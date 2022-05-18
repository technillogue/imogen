import random
from typing import Optional, Any
import torch
import tqdm
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from clip import clip
from core import ImgPrompt
import postnet
import subprocess

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        # m.half()
        # aka Glorot
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def clipboard(text: str) -> None:
    subprocess.run("fish -c clip", shell=True, input=text.encode())  # pylint: disable


printed = {}


def print_once(key: str, *args: Any) -> None:
    if key not in printed:
        print(*args)
    printed[key] = True


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

    def predict_text(self, text_embed: Tensor) -> Tensor:
        return self.net(self.narrow_projection(text_embed))

    def predict_wide(self, wide_embed: Tensor) -> Tensor:
        return self.net(self.wide_projection(wide_embed))

    forward = predict_text


def train(net: Optional[nn.Sequential], prompts: list[ImgPrompt]) -> nn.Sequential:
    writer = SummaryWriter() # comment=input("comment for run> "))  # type: ignore
    if not net:
        net = torch.load("reaction_predictor.pth").to(device) or Likely().to(device)
    # net.apply(init_weights)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = nn.L1Loss()
    epochs = 1
    batch_size = 4
    # data: list[tuple[Tensor, Tensor, float]] = []
    # for _ in range(epochs):
    #     for prompt in prompts:
    #         # it's [1, 512] normally
    #         prompt.embed = prompt.embed.to(device).to(torch.float32).reshape([512])
    #         for cutout in prompt.image_embed:
    #             # Tensor[512], Tensor[512], float
    #             data.append((prompt.embed, cutout.to(device), prompt.label))
    # random.shuffle(data)
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
        embeds = torch.cat([massage_embeds(prompt).unsqueeze(0) for prompt in batch])
        # embeds = (embeds - embeds.mean()) / embeds.std()
        actual = torch.cat([massage_actual(prompt).unsqueeze(0) for prompt in batch])
        prediction = net.predict_wide(embeds)  # pylint: disable
        # actual = torch.cat([Tensor([[label]]) for _, _, label in batch]).to(device)
        loss = loss_fn(prediction, actual)
        losses.append(float(loss))
        loss.mean().backward()
        opt.step()
        writer.add_scalar("loss/train", sum(losses) / len(losses), batch_index)  # type: ignore
        if (batch_index + 1) % 50 == 0:
            pbar.write(f"batch {batch_index} loss: {round(sum(losses)/len(losses), 4)}")
    writer.flush()  # type: ignore
    torch.save(net, "reaction_predictor_tuned.pth")
    print(f"train loss: {round(sum(losses) / len(losses), 4)}")
    return net


# typically Tensor[64, 1024], Tensor[64, 1]
def massage(prompt: ImgPrompt) -> tuple[Tensor, Tensor]:
    return massage_embeds(prompt), massage_actual(prompt)


def massage_embeds(prompt: ImgPrompt) -> Tensor:
    text = prompt.embed.to(device).reshape([512]).to(torch.float32).to(device)
    return torch.cat(
        [
            torch.cat([text, cutout]).unsqueeze(0)
            for cutout in prompt.image_embed.to(device)
        ]
    ).to(device)


def massage_actual(prompt: ImgPrompt) -> Tensor:
    return Tensor([[prompt.label] for _ in prompt.image_embed]).to(device)


def validate(prompts: list[ImgPrompt], net: Optional[nn.Module] = None) -> None:
    if not net:
        net = torch.load("reaction_predictor.pth").to(device)  # type: ignore
    assert net
    loss_fn = nn.L1Loss()
    losses = []
    messages = []
    for i, prompt in enumerate(prompts):
        prompt.embed = prompt.embed.reshape([512]).to(torch.float32).to(device)
        prediction = net.predict_wide(massage_embeds(prompt).unsqueeze(0)).to("cpu")
        actual = massage_actual(prompt).to("cpu").unsqueeze(0)
        if i < 20:
            messages.append(
                f"predicted: {round(float(prediction.mean()), 4)}, actual: {prompt.label} ({prompt.reacts}). {prompt.prompt}"
            )
        loss = loss_fn(prediction, actual)
        losses.append(float(loss.mean()))
    test_loss = f"test loss: {round(sum(losses) / len(losses), 4)}"
    clipboard(test_loss)
    print()
    print("\n".join(messages))


# base arch: train 0.01 test 0.4614
# 1024 epoch 5 batch 20
# train loss:  0.0208
# test loss: 0.4303
# revert 1024: 0.4562
# batch 64
# train loss: 0.0238
# test: 0.4031
# single image batchs
# train loss: 0.3053
# test loss: 0.375
# batches of 4 images, cutouts together
# train loss: 0.2831
# test loss: 0.4634
# epoch 20
# test loss: 0.4025
# batch 1 epoch 20
# train 0.15, test loss: 0.4259
# try revert to single image batch:
# train loss: 0.2768
# test loss: 0.4505
# make batches [batch_size, 64, 1024] instead of [batch_size * 64, 2014]
# train loss: 0.1522
# test loss: 0.4218
# standardize values to mean 0 stdev 1
# train loss: 0.1525
# test loss: 0.4377
# less neuron
# train loss: 0.18
# test loss: 0.4386
# transfer learning....
# train loss:  0.2979
# test loss: 0.4245
# transfer without saving
# train loss: 0.1837
# test loss: 0.4572
# 

def main() -> None:
    # net = Likely().to(device)
    net = torch.load("reaction_predictor_tuned.pth").to(device)
    net = postnet.main(net)
    prompts = torch.load("img_prompts.pth")  # type: ignore
    valid = len(prompts) // 5  # 20%
    train_set, valid_set = torch.utils.data.random_split(
        prompts, [len(prompts) - valid, valid]
    )
    print(len(train_set), len(valid_set))
    net = train(net, list(train_set))
    validate(list(valid_set), net)


def train_prod() -> None:
    prompts = torch.load("img_prompts.pth")  # type: ignore
    for prompt in prompts:
        prompt.embed = prompt.embed.to(device).to(torch.float32)
    train(prompts)


if __name__ == "__main__":
    main()
