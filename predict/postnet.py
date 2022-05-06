from typing import Optional
import random
import torch
import tqdm
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter

from get_embeddings import Prompt

device = "cpu"


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
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
        nn.Linear(256, 1),
        nn.Dropout(p=0.1),
    ).to(device)
    net.apply(init_weights)
    writer = SummaryWriter()  # type: ignore
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()
    epochs = 100
    batch_size = 50
    # 13000 / batch_size * epochs = 20k?
    prompts = prompts * epochs
    random.shuffle(prompts)
    batches = int(len(prompts) / batch_size)
    for batch_index in range(batches):
        opt.zero_grad()
        batch = prompts[
            batch_index * batch_size : batch_index * batch_size + batch_size
        ]
        embeds = torch.cat([prompt.embed.reshape([1, 512]) for prompt in batch])
        prediction = net(embeds)
        actual = torch.cat([Tensor([[float(bool(prompt.reacts))]]) for prompt in batch])
        loss = loss_fn(prediction, actual)
        loss.backward()
        opt.step()
        writer.add_scalar("loss/train", float(loss), batch_index)  # type: ignore
        if (batch_index + 1) % 10  == 0:
            print(f"batch {batch_index} loss: {float(loss)}")
    writer.flush()  # type: ignore
    torch.save(net, "reaction_predictor.pth")
    return net


def validate(prompts: list[Prompt], net: Optional[nn.Module] = None) -> None:
    if not net:
        net = torch.load("reaction_predictor.pth")  # type: ignore
    assert net
    loss_fn = nn.L1Loss()
    losses = []
    for i, prompt in enumerate(prompts):
        prediction = net(prompt.embed)
        actual = Tensor([float(bool(prompt.reacts))])
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


def main() -> None:
    prompts = torch.load("prompts.pth")  # type: ignore
    train_set = []
    valid_set = []
    for i, prompt in enumerate(prompts):
        prompt.embed = prompt.embed.reshape([512])
        if i % 10 < 8:
            train_set.append(prompt)
        else:
            valid_set.append(prompt)
    print(len(train_set), len(valid_set))
    net = train(list(train_set))
    validate(list(valid_set), net)


if __name__ == "__main__":
    main()
