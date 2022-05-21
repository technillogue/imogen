import random
import statistics
from typing import Any, Optional, Union, NewType, TypeVar
import torch
import tqdm
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
import postnet
import v2postnet
from core import ImgPrompt, EmbedPrompt, Prompt
from v2postnet import massage_actual, massage_embeds, print_once, clipboard

device = "cuda:0" if torch.cuda.is_available() else "cpu"

ClipEmbed = NewType("ClipEmbed", Tensor)  # can basically be [512], [batch_size, 512]
WideEmbed = NewType("WideEmbed", Tensor)  # [64, 1024] or [batch_size, 64, 1024]
Scalar = NewType("Scalar", Tensor)  # [1]
Cutout = NewType("Cutout", Tensor)  # [64, 512]


class MultiheadedSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=8,
        attn_dropout=0.01,
        proj_dropout=0.01,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dim must be divisible by number of heads."
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(
                2,  # qkv
                0,  # batch
                3,  # channel
                1,  # num_heads
                4,  # embed_dim
            )
        )
        q, k, v = torch.chunk(qkv, 3)

        attn = torch.bmm(q, k.transpose(-2, -1)) * self.scale  # <q,k> / sqrt(d)
        attn.softmax(dim=-1)  # Softmax over embedding dim
        attn = self.attn_dropout(attn)

        x = torch.bmm(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.projection(x)
        x = self.proj_dropout(x)

        return x


class Likely(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wide_projection = nn.Linear(1024, 512)
        self.narrow_projection = nn.Linear(512, 512)
        self.net = nn.Sequential(
            nn.ReLU(),
            # nn.LayerNorm(512),
            nn.Dropout(p=0.05),
            nn.Linear(512, 256),  # fc2
            nn.ReLU(),
            nn.Dropout(p=0.05),
            MultiheadedSelfAttention(256),
            # nn.Dropout(p=0.05),
            # nn.Linear(512, 256),  # fc3_256
            # nn.ReLU(),
            # nn.Dropout(p=0.1),
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
        img_batches = self.prepare_batches(
            img_prompts,  # 651
            batch_size=2,  # 256
            epochs=30,  #
        )
        text_batches = self.prepare_batches(
            text_prompts,  # 11720
            batch_size=32,
            epochs=12,  # 10577/40 * epochs = 2644
        )
        print("image batches:", len(img_batches), "text batches:", len(text_batches))
        mixed_batches = img_batches + text_batches
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
        print_once("imgemb", "img embed:", embeds.shape)
        print_once("predshape", "img prediction shape:", prediction.shape)
        print_once("actshape", "img actual shape:", actual.shape)
        # actual = torch.cat([Tensor([[label]]) for _, _, label in batch]).to(device)
        loss = self.loss_fn(prediction, actual)
        print_once("loss", "img loss: ", loss)
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
        print_once("txtemb", "text embed:", embeds.shape)
        actual = torch.cat([Tensor([[prompt.label]]) for prompt in batch]).to(device)
        loss = self.loss_fn(prediction, actual)
        return loss

    def train(self, batches: list[Union[list[ImgPrompt], list[Prompt]]]) -> Likely:
        # opt = torch.optim.SGD(self.net.parameters(), lr=1e-5, weight_decay=0.02)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-5)
        img_losses = []
        text_losses = []
        losses = []
        writer = SummaryWriter(
            comment=COMMENT
        )  # comment=input("comment for run> "))  # type: ignore
        pbar = tqdm.tqdm(batches, desc="batch")
        for i, batch in enumerate(pbar):
            if isinstance(batch[0], ImgPrompt):
                loss = self.train_img(batch)  # .mean?
                img_losses.append(loss)
                writer.add_scalar("loss/img", sum(img_losses) / len(img_losses), i)
            else:
                loss = self.train_text(batch)
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
# baseline with adam, dropout 0.05, img batch 4 epoch 15, text batch 32 epoch 6
# overall train loss:  0.3076
# text validation: test loss: 0.4353
# test loss: 0.4638
# 3x dropout 0.05
# 0.4253
# maybe 0.4418?
# no fixed bias
# 0.4604, 0.4338, 0.4056
# txt batch 16 epoch 3
# 0.4582, 0.4271, 0.3968
# okay, testing baseline with 5 runs, avg 0.43586 stdev 0.029 median 0.4441 range 0.3973-0.4682
# project to 1024
# mean 0.4402 stdev 0.03463, min 0.3954
# img batch 2
# mean: 0.4252 stdev: 0.0261 <------------ best
# txt epoch 12
# mean: 0.4388 stdev: 0.033 min: 0.3991
# txt epoch 6 batch 16
# mean: 0.4541 stdev: 0.028 min: 0.4249
# baseline, AdamW,
# mean: 0.4457 stdev: 0.0472 min: 0.409
# double text
# mean: 0.4391 stdev: 0.0221 min: 0.4062
# no_wide:
# mean: 0.4382 stdev: 0.0176 min: 0.4116
# fc2 256:
# mean: 0.4407 stdev: 0.0208 min: 0.4121
# img epoch 7:
# mean: 0.4563 stdev: 0.0469 min: 0.389
# GELU, img epoch 7
# mean: 0.433 stdev: 0.0218 min: 0.4107
# GELU, img epoch 15, lr1e-4:
# mean: 0.503 stdev: 0.0486 min: 0.4397
# AdamW lr 5e-5:
# mean: 0.4742 stdev: 0.0412 min: 0.4238
# Gelu img epoch 15 Adam 1e-5:
# mean: 0.4558 stdev: 0.0377 min: 0.398
# apparently baseline with img batch 2 epoch 15 txt batch 32 epoch 6 is now
# mean: 0.4618 stdev: 0.0083 min: 0.454
# reroll baseline:
# mean: 0.4565 stdev: 0.0195 min: 0.4299
# attention?! double batch
# 0.419
# 8 heads
# mean: 0.4097 stdev: 0.0028 min: 0.4066
# two transformers?
# mean: 0.4184 stdev: 0.005 min: 0.413

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
    return v2postnet.validate(list(img_valid_set), trainer.net)
    # return trainer


COMMENT = input("comment for run> ")
test_losses = [main() for i in tqdm.trange(5, desc="runs")]
stats = {
    "mean": statistics.mean(test_losses),
    "stdev": statistics.stdev(test_losses),
    "min": min(test_losses),
}
msg = COMMENT + ":\n" + " ".join(f"{k}: {round(v, 4)}" for k, v in stats.items())
print(msg)
clipboard(msg)
print("\a")  # bell
