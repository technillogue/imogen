import random
from typing import Any, Optional, Union, NewType, TypeVar
import torch
import tqdm
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
import postnet
import v2postnet
from core import ImgPrompt, EmbedPrompt, Prompt
from v2postnet import massage_actual, massage_embeds, print_once

device = "cuda:0" if torch.cuda.is_available() else "cpu"

ClipEmbed = NewType("ClipEmbed", Tensor)  # can basically be [512], [batch_size, 512]
WideEmbed = NewType("WideEmbed", Tensor)  # [64, 1024] or [batch_size, 64, 1024]
Scalar = NewType("Scalar", Tensor)  # [1]
Cutout = NewType("Cutout", Tensor)  # [64, 512]


class Likely(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wide_projection = nn.Linear(1024, 512)
        self.narrow_projection = nn.Linear(512, 512)
        self.net = nn.Sequential(
            nn.ReLU(),
#            nn.LayerNorm(512),
            nn.Linear(512, 512),  # fc2
            nn.ReLU(),
            nn.Linear(512, 256),  # fc3_256
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(256, 1),  # fc4_1
            nn.Sigmoid(),
        ).to(device)
        self.apply(postnet.init_weights)

    def predict_text(self, text_embed: ClipEmbed) -> Scalar:
        return self.net(self.narrow_projection(text_embed))

    def predict_wide(self, wide_embed: WideEmbed) -> Scalar:
        """
        wide_embed: Tensor[1024], [batch_size, 1024]
        """
        return self.net(self.wide_projection(wide_embed))

    forward = predict_text


class LikelyTrainer:
    def __init__(self) -> None:
        # net = torch.load("reaction_predictor_tuned.pth").to(device)
        self.net = Likely().to(device)
        self.loss_fn = nn.L1Loss()

    # could be  Union[list[ImgPrompt], list[Prompt]]]
    def prepare_batches(
        self, prompts: list[EmbedPrompt], batch_size: int = 10, epochs: int = 10
    ) -> list[list[EmbedPrompt]]:
        # preheat
        for prompt in prompts:
            prompt.embed = prompt.embed.to(device).to(torch.float32)
        data = [prompt for epoch in range(epochs) for prompt in prompts]
        random.shuffle(data)
        # we want to iterate through the indexes, taking the index of every e.g. 10th datapoint
        batches = [
            data[batch_start : batch_start + batch_size]
            for batch_start in range(0, len(data), batch_size)
        ]
        return batches

    def prepare_mixed_batches(
        self, img_prompts: list[ImgPrompt], text_prompts: list[Prompt]
    ) -> list[Union[list[ImgPrompt], list[Prompt]]]:
        mixed_batches = self.prepare_batches(
            img_prompts,  # 651
            batch_size=4,  # 256
            epochs=15,  # 
        ) + self.prepare_batches(
            text_prompts, # 11720
            batch_size=32,
            epochs=6,  # 10577/40 * epochs = 2644
        )
        random.shuffle(mixed_batches)
        return [
            batch
            for batches in [
                # self.prepare_batches(text_prompts, batch_size=32, epochs=1),
                mixed_batches,
                # self.prepare_batches(img_prompts, batch_size=2, epochs=1),
            ]
            for batch in batches
        ]

    def prepare_interleaved_batches(
        self, img_prompts: list[ImgPrompt], text_prompts: list[Prompt]
    ) -> list[Union[list[ImgPrompt], list[Prompt]]]:
        epochs = 10
        return [
            batch
            for i in range(epochs)
            for stream in (text_prompts, img_prompts)
            for batch in self.prepare_batches(stream, epochs=1)
        ]
        # exactly list[list[Prompt], list[ImgPrompt], ...]

    def train_img(self, batch: list[ImgPrompt]) -> Scalar:
        no_wide = False
        if no_wide:
            embeds = torch.cat(
                [prompt.image_embed.to(device) for prompt in batch]
                + [prompt.embed.to(device) for prompt in batch]
            )
            actual = Tensor(
                [[prompt.label] for prompt in batch for _ in prompt.image_embed]
                + [[prompt.label] for prompt in batch]
            ).to(device)
            prediction = self.net.predict_text(embeds)
        else:
            embeds = torch.cat(
                [massage_embeds(prompt).unsqueeze(0) for prompt in batch]
            )
            # embeds = (embeds - embeds.mean()) / embeds.std()
            actual = torch.cat(
                [massage_actual(prompt).unsqueeze(0) for prompt in batch]
            )
            prediction = self.net.predict_wide(embeds)  # pylint: disable
        print_once("predshape", "img prediction shape:", prediction.shape)
        print_once("actshape", "img actual shape:", actual.shape)
        # actual = torch.cat([Tensor([[label]]) for _, _, label in batch]).to(device)
        loss = self.loss_fn(prediction, actual)
        print_once("loss", loss.shape)
        return loss

    def train_text(self, batch: list[Prompt]) -> Scalar:
        double = False
        if double:
            embeds = torch.cat(
                [torch.cat([prompt.embed, prompt.embed], dim=1) for prompt in batch]
            )
            prediction = self.net.predict_wide(embeds)
        else:
            embeds = torch.cat([prompt.embed for prompt in batch])
            prediction = self.net.predict_text(embeds)
        actual = torch.cat([Tensor([[prompt.label]]) for prompt in batch]).to(device)
        loss = self.loss_fn(prediction, actual)
        return loss

    def train(self, batches: list[Union[list[ImgPrompt], list[Prompt]]]) -> Likely:
        # opt = torch.optim.SGD(self.net.parameters(), lr=1e-5, weight_decay=0.02)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-5)
        img_losses = []
        text_losses = []
        losses = []
        writer = SummaryWriter(comment=input("comment for run> "))  # type: ignore
        pbar = tqdm.tqdm(batches)
        for i, batch in enumerate(pbar):
            if isinstance(batch[0], ImgPrompt):
                loss = self.train_img(batch)  # .mean?
                print_once("imgshape", "image loss shape:", loss.shape)
                img_losses.append(loss)
                writer.add_scalar("loss/img", sum(img_losses) / len(img_losses), i)
            else:
                loss = self.train_text(batch)
                print_once("txtshape", "text loss shape:", loss.shape)
                text_losses.append(loss)
                writer.add_scalar("loss/text", sum(text_losses) / len(text_losses), i)
            losses.append(float(loss))
            print_once("step loss", "step loss: ", loss)
            loss.backward()
            opt.step()
            writer.add_scalar("loss/train", sum(losses) / len(losses), i)  # type: ignore
            if (i + 1) % 50 == 0:
                pbar.write(f"batch {i} loss: {round(sum(losses)/len(losses), 4)}")

        writer.flush()  # type: ignore
        print("overall train loss: ", round(sum(losses) / len(losses), 4))
        torch.save(self.net, "likely.pth")
        return self.net


# baseline with sandwiched mixed img epoch 10 batch 2 text epoch 5 batch 32
# test loss: 0.4774
# trying just images, didn't converge with dropout but removing one of them worked
# test loss: 0.4516

# batch 8
# overall train loss:  0.2945
# test loss: 0.4239
# ....
# batch 8, no layernorm, 1e-5, 15 epochs
# test loss: 0.4251
# add epoch 3 batch 32 text with double
# overall train loss:  0.3569
# test loss: 0.4158 (0.5 txt)
# don't double, use predict_text
# overall train loss:  0.3358
# text test loss: 0.4151
# img test loss: 0.3871

# SGD 1e-5 ..
# SDG 1e-4, weight_decay 0.01, img batch 8 epoch 15 txt batch 32 epoch 3
# test loss: 0.4591
# epochs * 50%
# test loss: 0.47

# new dataset with cutn 128
# AdamW 1e-5, img batch 4 epoch 4 txt batch 512 epoch 28
# test loss: 0.4281
# img epoch 15 txt batch 32 epoch 3
# test loss: 0.4621 but better train convergence 
# img batch 8 and 2 are both worse than 4; 8 doesn't converge and 2 overfits 
# layernorm, dropout 0.1
# 0.4209
# no layernorm, dropout 0.41
# realize we were keeping dropout for validation and that made things... better?
# sgd lr 1e-5 decay 0.02
# 0.4576
# 
def main():
    ## set up text
    text_prompts = torch.load("text_prompts.pth")  # type: ignore
    text_valid = len(text_prompts) // 10
    text_train_set, text_valid_set = torch.utils.data.random_split(
        text_prompts, [len(text_prompts) - text_valid, text_valid]
    )
    print(len(text_train_set), len(text_valid_set))

    ## set up images

    img_prompts = torch.load("img_prompts.pth")  # type: ignore
    img_valid = len(img_prompts) // 5  # 20%
    img_train_set, img_valid_set = torch.utils.data.random_split(
        img_prompts, [len(img_prompts) - img_valid, img_valid]
    )
    print("img", len(img_train_set), len(img_valid_set))

    trainer = LikelyTrainer()
    batches = trainer.prepare_mixed_batches(img_train_set, text_train_set)
    trainer.train(batches)
    trainer.net.eval()
    print("text validation:")
    postnet.validate(list(text_valid_set), trainer.net)
    print("image validation")
    v2postnet.validate(list(img_valid_set), trainer.net)
    return trainer


result = main()