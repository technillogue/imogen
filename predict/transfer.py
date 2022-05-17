import random
from typing import Any, Optional
import torch
import tqdm
from clip import clip
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
import postnet
import v2postnet
from core import ImgPrompt, EmbedPrompt, Prompt
from v2postnet import massage_actual, massage_embeds, print_once

device = "cuda:0" if torch.cuda.is_available() else "cpu"


ClipEmbed = NewType("[..., 512]", Tensor)
WideEmbed = newType("[..., 1024]", Tensor)
Scalar = NewType("[1]", Tensor)
Cutout = NewType("[64, 512]", Tensor)


class Likely(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wide_projection = nn.Linear(1024, 512)
        self.narrow_projection = nn.Linear(512, 512)
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512),  # fc2
            nn.ReLU(),
            nn.Linear(512, 256),  # fc3_256
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1),  # fc4_1
            nn.Sigmoid(),
        ).to(device)
        self.apply(init_weights)

    def predict_text(self, text_embed: ClipEmbed) -> Scalar:
        return self.net(self.narrow_projection(text_embed))

    def predict_wide(self, wide_embed: WideEmbed) -> Scalar:
        """
        wide_embed: Tensor[1024], [batch_size, 1024]
        """
        return self.net(self.wide_projection(wide_embed))

    forward = predict_text


class LikelyTrainer:
    pass
    # opt,
    def __init__(self) -> None:
        # net = torch.load("reaction_predictor_tuned.pth").to(device)
        self.net = Likely().to(device)
        self.loss_fn = nn.L1Loss()

    def train_img(self, batch: list[ImgPrompt]) -> Scalar:
        embeds = torch.cat([massage_embeds(prompt).unsqueeze(0) for prompt in batch])
        # embeds = (embeds - embeds.mean()) / embeds.std()
        actual = torch.cat([massage_actual(prompt).unsqueeze(0) for prompt in batch])
        prediction = self.net.predict_wide(embeds)  # pylint: disable
        # actual = torch.cat([Tensor([[label]]) for _, _, label in batch]).to(device)
        loss = self.loss_fn(prediction, actual)
        print_once("loss", loss.shape)
        return loss

    def train_text(self, batch: list[Prompt]) -> Scalar:
        embeds = torch.cat([prompt.embed for prompt in batch])
        prediction = self.net.predict_text(embeds)
        actual = torch.cat([Tensor([[prompt.label]]) for prompt in batch]).to(device)
        loss = self.loss_fn(prediction, actual)
        return loss

    #        Union[list[ImgPrompt], list[Prompt]]]
    def prepare_batches(
        self, inp: list[EmbedPrompt], batch_size: int = 10, epochs: int = 10
    ) -> list[list[EmbedPrompt]]:
        # for prompt in prompts:
        #     prompt.embed = prompt.embed.to(device).to(torch.float32)
        data = [prompt for epoch in range(epochs) for prompt in prompts]
        random.shuffle(data)
        batch_count = int(len(prompts) / batch_size)
        # we want to iterate through the indexes, taking the index of every e.g. 10th datapoint
        batches = [
            data[batch_start : batch_start + batch_size]
            for batch_start in range(0, len(data), batch_size)
        ]
        return batches

    def prepare_mixed_batches(
        self, img_prompts: list[ImgPrompt], text_prompts: list[Prompt]
    ) -> list[Union[list[ImgPrompt], list[Prompt]]]:
        all_batches = prepare_batches(img_prompts) + prepare_batches(text_prompts)
        random.shuffle(all_batches)
        return all_batches

    def prepare_interleaved_batches(
        self, img_prompts: list[ImgPrompt], text_prompts: list[Prompt]
    ) -> list[Union[list[ImgPrompt], list[Prompt]]]:
        epochs = 10
        return [
            batch
            for i in range(epochs)
            for stream in (text_prompts, img_prompts)
            for batch in prepare_batches(stream, epochs=1)
        ]
        # exactly list[list[Prompt], list[ImgPrompt], ...]

    def train(self, batches: list[Union[list[ImgPrompt], list[Prompt]]]) -> Scalar:
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        losses = []
        writer = SummaryWriter()  # comment=input("comment for run> "))  # type: ignore
        pbar = tqdm.tqdm(batches)
        for batch in pbar:
            if isinstance(batch[0], ImgPrompt):
                loss = self.train_img(batch)  # .mean?
                print_once("imgshape", loss.shape)
            else:
                loss = self.train_text(batch)
            losses.append(float(loss))
            loss.mean().backward()
            opt.step()
            writer.add_scalar("loss/train", sum(all_loss) / len(all_loss), len(all_loss))  # type: ignore
            if (batch_index + 1) % 50 == 0:
                pbar.write(
                    f"batch {batch_index} loss: {round(sum(losses)/len(losses), 4)}"
                )

    def main(self) -> None:
        self.prepare_mixed_batches(...)


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


net = evil_train(list(img_train_set), list(text_train_set))
print("text validation:")
postnet.validate(list(text_valid_set), trainer.net)
print("image validation")
v2postnet.validate(list(img_valid_set), trainer.net)

def evil_train(img_prompts, text_prompts):
    # if not net:
    #     net = torch.load("reaction_predictor.pth").to(device) or Likely().to(device)
    loss_fn = nn.L1Loss()
    all_loss = []

    def train_text(prompts=text_train_set):
        epochs = 1
        batch_size = 10
        for prompt in prompts:
            prompt.embed = prompt.embed.to(device).to(torch.float32)
        data = [prompt for epoch in range(epochs) for prompt in prompts]
        random.shuffle(data)
        batch_count = int(len(prompts) / batch_size)
        # we want to iterate through the indexes, taking the index of every e.g. 10th datapoint
        range(0, len(data), batch_size)
        [
            data[batch_start : batch_start + batch_size]
            for batch_start in range(0, len(data), batch_size)
        ]
        len_data = len(data)

        prompts = prompts * epochs
        random.shuffle(prompts)
        batches = int(len(prompts) / batch_size)
        losses = []
        pbar = tqdm.trange(batches)
        for batch_index in pbar:
            opt.zero_grad()
            batch = prompts[
                batch_index * batch_size : batch_index * batch_size + batch_size
            ]
            embeds = torch.cat([prompt.embed for prompt in batch])
            prediction = net(embeds)
            actual = torch.cat([Tensor([[prompt.label]]) for prompt in batch]).to(
                device
            )
            loss = loss_fn(prediction, actual)
            all_loss.append(float(loss))
            losses.append(float(loss))
            loss.backward()
            opt.step()
            writer.add_scalar("loss/train", sum(all_loss) / len(all_loss), len(all_loss))  # type: ignore
            if (batch_index + 1) % 100 == 0:
                pbar.write(
                    f"batch {batch_index} loss: {round(sum(losses)/len(losses), 4)}"
                )
        print("text train loss: ", round(sum(losses) / len(losses), 4))

    def train_image(prompts=img_train_set):
        epochs = 1
        batch_size = 4
        for prompt in prompts:
            prompt.embed = (
                prompt.embed.reshape([512]).to(torch.float32).to(device)
            )  # pylint: disable
        prompts = prompts * epochs
        random.shuffle(prompts)
        batches = int(len(prompts) / batch_size)
        losses = []
        pbar = tqdm.trange(batches)
        for batch_index in pbar:
            opt.zero_grad()
            batch_start = batch_index * batch_size
            batch = prompts[batch_start : batch_start + batch_size]

            embeds = torch.cat(
                [massage_embeds(prompt).unsqueeze(0) for prompt in batch]
            )
            # embeds = (embeds - embeds.mean()) / embeds.std()
            actual = torch.cat(
                [massage_actual(prompt).unsqueeze(0) for prompt in batch]
            )
            prediction = net.predict_wide(embeds)  # pylint: disable
            # actual = torch.cat([Tensor([[label]]) for _, _, label in batch]).to(device)
            loss = loss_fn(prediction, actual)
            losses.append(float(loss))
            all_loss.append(float(loss))
            print_once("loss", loss.shape)
            loss.mean().backward()
            opt.step()
            writer.add_scalar("loss/train", sum(all_loss) / len(all_loss), len(all_loss))  # type: ignore
            if (batch_index + 1) % 50 == 0:
                pbar.write(
                    f"batch {batch_index} loss: {round(sum(losses)/len(losses), 4)}"
                )
        print(f"image train loss: {round(sum(losses) / len(losses), 4)}")

    for i in range(10):
        train_text(text_prompts)
        train_image(img_prompts)
    writer.flush()  # type: ignore
    print("overall train loss: ", round(sum(all_loss) / len(all_loss), 4))
    torch.save(net, "reaction_predictor_tuned.pth")
    return net


# baseline
# 0.4458

