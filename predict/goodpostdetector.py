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
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class MyClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pront = True
        self.perceptor = clip.load("ViT-B/32", jit=False, device=device)[0]
        self.perceptor.eval()

    def forward(self, tokens: Tensor) -> Tensor:
        encoded = self.perceptor.encode_text(tokens)
        normed = nn.functional.normalize(encoded, dim=1)
        if self.pront:
            print(encoded)
            print(normed)
            self.pront = False
        return normed 


class GoodPostNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pront = True
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

    def forward(self, tokens: Tensor) -> Tensor:
        embed = self.perceptor(tokens)
        prediction = self.net(embed)
        if self.pront:
            print(embed)
            print(prediction)
            self.pront = False
        return prediction


def train_with_clip(prompts: list[Prompt]) -> nn.Sequential:
    writer = SummaryWriter()  # type: ignore
    for prompt in prompts:
        prompt.tokens = clip.tokenize(prompt.prompt, truncate=True).to(device)
    print("tokenized prompts")
    net = GoodPostNetwork()
    print("net instanciated")
    scaler = torch.cuda.amp.GradScaler()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = nn.L1Loss()
    epochs = 10
    batch_size = 10
    prompts = prompts * epochs
    random.shuffle(prompts)
    batch_count = int(len(prompts) / batch_size)
    losses = []
    print("starting training")
    for batch_index in range(batch_count):
        opt.zero_grad()
        batch = prompts[
            batch_index * batch_size : batch_index * batch_size + batch_size
        ]
        tokens = torch.cat([prompt.tokens for prompt in batch])
        with torch.cuda.amp.autocast():
            prediction = net(tokens)
            assert prediction.dtype == torch.float16
            actual = torch.cat([Tensor([[prompt.label]]) for prompt in batch]).to(device)
            loss = loss_fn(prediction, actual)
            assert loss.dtype == torch.float32
            losses.append(float(loss))
        scaler.scale(loss).backward()
        scaler.step(opt) # ValueError: Attempting to unscale FP16 gradients.
        scaler.update()
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
        actual = Tensor([prompt.label]).to(device)
        if i < 20:
            print(
                f"predicted: {round(float(prediction), 4)}, actual: {prompt.label} ({prompt.reacts}). {prompt.prompt}"
            )
        loss = loss_fn(prediction, actual)
        losses.append(float(loss))
    print(f"L1: {round(sum(losses) / len(losses), 4)}")


def main() -> None:
    prompts = torch.load("basic_prompts.pth")  # type: ignore
    train_set = []
    valid_set = []
    for i, prompt in enumerate(prompts):
        if i % 10 < 8:
            train_set.append(prompt)
        else:
            valid_set.append(prompt)
    print(len(train_set), len(valid_set))
    net = train_with_clip(list(train_set))
    validate_with_toks(list(valid_set), net)


if __name__ == "__main__":
    main()
