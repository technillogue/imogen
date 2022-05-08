import random
from typing import Optional
import torch
import tqdm
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from clip import clip
from core import Prompt
from torch.cuda.amp import autocast

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        # m.half()
        # aka Glorot
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train(prompts: list[Prompt]) -> nn.Sequential:
    net = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    ).to(device)
    net.apply(init_weights)
    writer = SummaryWriter()  # type: ignore
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = nn.L1Loss()
    epochs = 10
    batch_size = 10
    prompts = prompts * epochs
    random.shuffle(prompts)
    batches = int(len(prompts) / batch_size)
    losses = []
    for batch_index in range(batches):
        opt.zero_grad()
        batch = prompts[
            batch_index * batch_size : batch_index * batch_size + batch_size
        ]
        embeds = torch.cat([prompt.embed for prompt in batch])
        prediction = net(embeds)
        actual = torch.cat(
            [Tensor([[float(bool(prompt.reacts))]]) for prompt in batch]
        ).to(device)
        loss = loss_fn(prediction, actual)
        losses.append(float(loss))
        loss.backward()
        opt.step()
        writer.add_scalar("loss/train", sum(losses) / len(losses), batch_index)  # type: ignore
        if (batch_index + 1) % 100 == 0:
            print(f"batch {batch_index} loss: {round(sum(losses)/len(losses), 4)}")
    writer.flush()  # type: ignore
    torch.save(net, "reaction_predictor.pth")
    print("train loss: ", round(sum(losses) / len(losses), 4))
    return net


def validate(prompts: list[Prompt], net: Optional[nn.Module] = None) -> None:
    if not net:
        net = torch.load("reaction_predictor.pth").to(device)  # type: ignore
    assert net
    loss_fn = nn.L1Loss()
    losses = []
    messages = []
    for i, prompt in enumerate(prompts):
        prediction = net(prompt.embed)
        actual = (
            Tensor([float(bool(prompt.reacts))]).reshape(prediction.shape).to(device)
        )
        if i < 20:
            messages.append(
                f"predicted: {round(float(prediction), 4)}, actual: {prompt.label} ({prompt.reacts}). {prompt.prompt}"
            )
        loss = loss_fn(prediction, actual)
        losses.append(float(loss))
    print(f"test loss: {round(sum(losses) / len(losses), 4)}")
    print("\n".join(messages))


# MSE: 0.49840676848697946
# switch to 512-256-256-1
# ~0.25
# use full 13k over 11k
# MSE: 0.2588944340470567
# oops was validating with training set
# MSE: 0.2626
# switch to 512-512-512-256-1
# MSE: 0.2606
# xavier init, L1
# 0.42
# use running loss in indicators, tweaks
# 0.44
# iterate through data twice
# 0.4265
# dropout = 0.1 and lr 1e-3
# 0.4359
# lr 1e-4
# 0.4256
# epoch = 2
# L1: 0.4183
# epoch 10, batches of 100, shuffle
# L1: 0.44
# 512-512-256-1
# L1: 0.4426
# 100 epochs, lr 1e-3
# 0.4107
# 100 epochs, lr 1e-5
# 0.4531
# 100 epochs, batch 10, lr 1e-4
# 0.4264
# 100 epochs, batch 50, lr 1e-3
# L1: 0.4379
# 10 epochs, batch 1 (total average loss indicator)
# train: 0.36 L1: 0.4195
# add ReLU at the end, 20 epochs, batch 10
# >0.5
# 2 epochs, batch 1
# 0.4873
# sigmoid at the end, epochs = 10, batch = 10, lr 1e-4
# train 0.3551838362793262, validation 0.4069
# epochs = 100, batch = 50
# train 0.1826724065951373, 0.4225
# epochs = 2, batch = 1
# 0.439893, 0.4249
# epochs = 100, batch = 50, lr = 1e-5
# 0.374307, 0.4145
# epochs = 20, batch = 10 lr=1e-4
# loss: 0.37430778, L1: 0.4145
# epochs = 10, batch = 10, lr=1e-4
# loss: 0.357416698303858, L1: 0.4058
# after refactor, excluding prompts that were never sent and prompts with "/imagine" or "in line"
# train 0.330879863, test 0.3921
# Dropout -> Lin -> Sigmoid rather than Lin -> Dropout -> Sigmoid
# train loss:  0.3088, test loss: 0.3949

def main() -> None:
    prompts = torch.load("prompts.pth")  # type: ignore
    train_set = []
    valid_set = []
    for i, prompt in enumerate(prompts):
        prompt.embed = prompt.embed.to(device).to(torch.float32)
        if i % 10 < 8:
            train_set.append(prompt)
        else:
            valid_set.append(prompt)
    print(len(train_set), len(valid_set))
    net = train(list(train_set))
    validate(list(valid_set), net)


def train_prod() -> None:
    prompts = torch.load("prompts.pth")  # type: ignore
    for prompt in prompts:
        prompt.embed = prompt.embed.to(device).to(torch.float32)
    net = train(prompts)


if __name__ == "__main__":
    main()
