import dataclasses
import random
import torch as T
import tqdm
import torch
from torch import Tensor, nn


@dataclasses.dataclass
class Prompt:
    prompt: str
    reacts: int
    loss: float
    embed: Tensor


device = "cpu"

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
    nn.Dropout(p=0.1),
).to(device)


def train():
    prompts = torch.load("prompts.pth")  # 11k prompts
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = T.nn.MSELoss()
    progress_bar = tqdm.tqdm(enumerate(prompts[: int(len(prompts) * 0.8)]))
    for i, prompt in progress_bar:
        opt.zero_grad()
        prediction = net(prompt.embed)
        actual = T.Tensor([[[float(bool(prompt.reacts))]]])
        loss = loss_fn(prediction, actual)
        loss.backward()
        opt.step()
        if i % int(len(prompts) / 10) == 0:
            progress_bar.set_description(f"i {i} loss: {loss}")

    torch.save(net, "reaction_predictor.pth")


def validate():
    net = torch.load("reaction_predictor.pth")
    prompts = torch.load("prompts.pth")
    test = list(prompts[-int(len(prompts) * 0.8):])
    random.shuffle(test)
    loss_fn = T.nn.MSELoss()
    losses = []
    for i, prompt in enumerate(test):
        prediction = net(prompt.embed)
        actual = T.Tensor([[[float(bool(prompt.reacts))]]])
        if i < 20:
            print(
                f"predicted: {round(float(prediction), 4)}, actual: {int(bool(prompt.reacts))} ({prompt.reacts}). {prompt.prompt}"
            )
        loss = loss_fn(prediction, actual)
        losses.append(float(loss))
    print(f"MSE: {round(sum(losses) / len(losses), 4)}")


# MSE: 0.49840676848697946
# switch to 256
# ~0.25
# full 13k
# MSE: 0.2588944340470567
# oops was validating with training set 
# switch to 512-512-512-256-1
# MSE: 0.2626
# MSE: 0.2606
train()
validate()