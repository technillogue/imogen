import random
from typing import Optional
import torch
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from clip import clip
from core import Prompt
from torch.cuda.amp import autocast

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        # aka Glorot
        m.half()
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class MyClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.perceptor = clip.load("ViT-B/32", jit=False)[0]

    def forward(self, tokens: Tensor) -> Tensor:
        return nn.functional.normalize(self.perceptor.encode_text(tokens), dim=1)


class GoodPostNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Dropout(p=0.1),
            nn.Sigmoid(),
        ).to(device)
        self.net.apply(init_weights)
        self.perceptor = MyClip()

    # @autocast()
    def forward(self, tokens: Tensor) -> Tensor:
        return self.net(self.perceptor(tokens))


def train_with_clip(prompts: list[Prompt]) -> nn.Sequential:
    writer = SummaryWriter()  # type: ignore
    for prompt in prompts:
        prompt.tokens = clip.tokenize(prompt.prompt, truncate=True).to(device)
    net = GoodPostNetwork()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = nn.L1Loss()
    epochs = 10
    batch_size = 10
    # 13000 / batch_size * epochs = 20k?
    prompts = prompts * epochs
    random.shuffle(prompts)
    batches = int(len(prompts) / batch_size)
    losses = []
    for batch_index in range(batches):
        opt.zero_grad()
        batch = prompts[
            batch_index * batch_size : batch_index * batch_size + batch_size
        ]
        tokens = torch.cat([prompt.tokens for prompt in batch])
        prediction = net(tokens)
        actual = torch.cat([Tensor([[prompt.label]]) for prompt in batch]).to(device)
        loss = loss_fn(prediction, actual)
        losses.append(float(loss))
        loss.backward()
        opt.step()
        writer.add_scalar("loss/train", sum(losses) / len(losses), batch_index)  # type: ignore
        if (batch_index + 1) % 100 == 0:
            print(f"batch {batch_index} loss: {sum(losses)/len(losses)}")
    writer.flush()  # type: ignore
    torch.save(net, "reaction_predictor.pth")
    return net


def validate_with_toks(prompts: list[Prompt], net: Optional[nn.Module] = None) -> None:
    if not net:
        net = torch.load("reaction_predictor.pth").to(device)  # type: ignore
    assert net
    loss_fn = nn.L1Loss()
    losses = []
    for i, prompt in enumerate(prompts):
        tokens = clip.tokenize(prompt.prompt, truncate=True).to(device)
        prediction = net(tokens)
        actual = Tensor([float(bool(prompt.reacts))]).to(device)
        if i < 20:
            print(
                f"predicted: {round(float(prediction), 4)}, actual: {int(bool(prompt.reacts))} ({prompt.reacts}). {prompt.prompt}"
            )
        loss = loss_fn(prediction, actual)
        losses.append(float(loss))
    print(f"L1: {round(sum(losses) / len(losses), 4)}")


def main() -> None:
    prompts = torch.load("basic_prompts.pth")  # type: ignore
    train_set = []
    valid_set = []
    for i, prompt in enumerate(prompts):
        # prompt.embed = prompt.embed.to(device)
        if i % 10 < 8:
            train_set.append(prompt)
        else:
            valid_set.append(prompt)
    print(len(train_set), len(valid_set))
    net = train_with_clip(list(train_set))
    validate_with_toks(list(valid_set), net)


if __name__ == "__main__":
    main()
