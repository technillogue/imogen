import random
from typing import Optional
import torch
import tqdm
from clip import clip
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from core import TokenPrompt

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
        self.perceptor.float()

    def forward(self, tokens: Tensor) -> Tensor:
        encoded = self.perceptor.encode_text(tokens)
        if self.pront:
            print(encoded)
            self.pront = False
        return nn.functional.normalize(encoded, dim=1)


class GoodPostNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pront = True
        self.net = nn.Sequential(
            nn.Linear(512, 512),  # fc1
            nn.ReLU(),
            nn.Linear(512, 256),  # fc2
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1),  # fc2_1
            nn.Sigmoid(),
        ).to(device)
        self.net.apply(init_weights)
        self.perceptor = MyClip()

    def forward(self, tokens: Tensor) -> Tensor:
        embed = self.perceptor(tokens)
        prediction = self.net(embed.to(torch.float32))
        if self.pront:
            print(embed)
            print(prediction)
            self.pront = False
        return prediction


def train_with_clip(prompts: list[TokenPrompt]) -> GoodPostNetwork:
    writer = SummaryWriter()  # type: ignore
    for prompt in tqdm.tqdm(prompts, desc="moving tokens to device"):
        prompt.tokens = prompt.tokens.to(device)
    net = GoodPostNetwork()
    print("net instanciated")
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-5)
    loss_fn = nn.L1Loss()
    epochs = 1
    batch_size = 10
    prompts = prompts * epochs
    random.shuffle(prompts)
    batch_count = int(len(prompts) / batch_size)
    losses = []
    print("starting training")
    pbar = tqdm.trange(batch_count)
    for batch_index in pbar:
        opt.zero_grad()
        batch = prompts[
            batch_index * batch_size : batch_index * batch_size + batch_size
        ]
        tokens = torch.cat([prompt.tokens for prompt in batch])
        with torch.cuda.amp.autocast(enabled=False):
            prediction = net(tokens)
            actual = torch.cat([Tensor([[prompt.label]]) for prompt in batch]).to(
                device
            )
            loss = loss_fn(prediction, actual)
            losses.append(float(loss))
        scaler.scale(loss).backward()
        scaler.step(opt)  # ValueError: Attempting to unscale FP16 gradients.
        scaler.update()
        writer.add_scalar("loss/train", sum(losses) / len(losses), batch_index)  # type: ignore
        if (batch_index + 1) % 100 == 0:
            pbar.write(f"batch {batch_index} loss: {round(sum(losses)/len(losses), 4)}")
    writer.flush()  # type: ignore
    torch.save(net, "reaction_predictor_clip.pth")
    return net


def validate_with_toks(
    prompts: list[TokenPrompt], net: Optional[nn.Module] = None
) -> None:
    if not net:
        net = torch.load("reaction_predictor_clip.pth").to(device)  # type: ignore
    assert net
    loss_fn = nn.L1Loss()
    losses = []
    messages = []
    for i, prompt in enumerate(prompts):
        prediction = net(prompt.tokens.to(device))
        actual = Tensor([prompt.label]).to(device).reshape(prediction.shape)
        if i < 20:
            messages.append(
                f"predicted: {round(float(prediction), 4)}, actual: {prompt.label} ({prompt.reacts}). {prompt.prompt}"
            )
        loss = loss_fn(prediction, actual)
        losses.append(float(loss))
    print(f"L1: {round(sum(losses) / len(losses), 4)}")
    print("\n".join(messages))


# lr 1e-4 batch=10 epoch=10
# train 0.4976, test: 0.494
# lr 5e-5 batch=100 epochs=100
# train 0.4997, test 0.4938. seems to predict 0 or 0.5 for everything
# move dropout, layernorm at the front,  epoch 1 batch 10, same worsening loss, predicts close to 1 for everything
# ...lots of attempts that also don't converge...
# lr 1e-5 batch 10 epoch 1 batchnorm1d converges with 0.447, but can't validate (Expected more than 1 value per channel when training, got input size torch.Size([1, 512]))
# no batchnorm or layernorm but do normalize embeddings (maybe the same...?)


def main() -> None:
    prompts = torch.load("token_prompts.pth")  # type: ignore
    valid = len(prompts) // 5
    train_set, valid_set = torch.utils.data.random_split(
        prompts, [len(prompts) - valid, valid]
    )
    print(len(train_set), len(valid_set))
    net = train_with_clip(list(train_set))
    validate_with_toks(list(valid_set), net)


if __name__ == "__main__":
    main()
