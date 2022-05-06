from typing import Optional
import dataclasses
import random
import torch
import tqdm
from torch import Tensor, nn
from get_embeddings import Prompt

from torch.utils.tensorboard import SummaryWriter


device = "cpu"


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


net = nn.Sequential(
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
    nn.Dropout(p=0.2),
).to(device)

net.apply(init_weights)


def train(prompts: list[Prompt]) -> nn.Module:
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = nn.L1Loss()
    progress_bar = tqdm.tqdm(enumerate(prompts))

    writer = SummaryWriter()
    current_loss_batch = []
    for i, prompt in progress_bar:
        opt.zero_grad()
        prediction = net(prompt.embed)
        actual = Tensor([[[float(bool(prompt.reacts))]]])
        loss = loss_fn(prediction, actual)
        current_loss_batch.append(float(loss))
        loss.backward()
        opt.step()
        if (i + 1) % int(len(prompts) / 10) == 0:
            avg_loss = sum(current_loss_batch) / len(current_loss_batch)
            writer.add_scalar("Loss/train", avg_loss, i)
            print(f"i {i} loss: {avg_loss}")
    writer.flush()
    torch.save(net, "reaction_predictor.pth")
    return net


def validate(prompts: list[Prompt], net: Optional[nn.Module] = None) -> None:
    if not net:
        net = torch.load("reaction_predictor.pth")  # type: ignore
    loss_fn = nn.L1Loss()
    losses = []
    for i, prompt in enumerate(prompts):
        prediction = net(prompt.embed)
        actual = Tensor([[[float(bool(prompt.reacts))]]])
        if i < 20:
            print(
                f"predicted: {round(float(prediction), 4)}, actual: {int(bool(prompt.reacts))} ({prompt.reacts}). {prompt.prompt}"
            )
        loss = loss_fn(prediction, actual)
        losses.append(float(loss))
    print(f"L1: {round(sum(losses) / len(losses), 4)}")


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
# use batch loss in indicators, tweaks
# 0.44
if __name__ == "__main__":
    prompts = torch.load("prompts.pth")  # type: ignore
    train_set = []
    valid_set = []
    for i, prompt in enumerate(prompts):
        if i % 10 < 8:
            train_set.append(prompt)
        else:
            valid_set.append(prompt)
    print(len(train_set), len(valid_set))
    net = train(list(train_set))
    validate(list(valid_set), net)
