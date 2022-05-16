import random
from typing import Optional
import torch
import tqdm
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from clip import clip
from core import ImgPrompt
import subprocess

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        # m.half()
        # aka Glorot
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def clipboard(text: str) -> None:
    subprocess.run("fish -c clip", shell=True, input=text.encode()) # pylint: disable

def train(prompts: list[ImgPrompt]) -> nn.Sequential:
    writer = SummaryWriter(comment=input("comment for run> "))  # type: ignore
    net = nn.Sequential(
        nn.Linear(1024, 1024),  # fc1
        nn.ReLU(),
        nn.LayerNorm(1024),
        nn.Linear(1024, 512),  # fc2
        nn.ReLU(),
        nn.Linear(512, 256),  # fc3_256
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(256, 1),  # fc4_1
        nn.Sigmoid(),
    ).to(device)
    net.apply(init_weights)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = nn.L1Loss()
    epochs = 5
    batch_size = 64
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
        prompt.embed = prompt.embed.reshape([512]).to(torch.float32).to(device) # pylint: disable
    prompts = prompts * epochs
    random.shuffle(prompts)
    # batches = int(len(data) / batch_size)
    losses = []
    # pbar = tqdm.trange(batches)
    # for batch_index in pbar:
    pbar = tqdm.tqdm(enumerate(prompts))
    for i, prompt in pbar:
        opt.zero_grad()
        # batch_start = batch_index * batch_size
        # batch = data[batch_start : batch_start + batch_size]
        # # Tensor[batch_size, 1024], unsqueeze adds a dimension to concat along
        # embeds = torch.cat(
        #     [torch.cat([text, img]).unsqueeze(0) for text, img, _ in batch]
        # )
        embeds, actual = massage(prompt)
        prediction = net(embeds) # pylint: disable
        # actual = torch.cat([Tensor([[label]]) for _, _, label in batch]).to(device)
        loss = loss_fn(prediction, actual)
        losses.append(float(loss))
        loss.mean().backward()
        opt.step()
        writer.add_scalar("loss/train", sum(losses) / len(losses), i)  # type: ignore
        if (i + 1) % 50 == 0:
            pbar.write(f"batch {i} loss: {round(sum(losses)/len(losses), 4)}")
    writer.flush()  # type: ignore
    torch.save(net, "reaction_predictor.pth")
    print(f"train loss: {round(sum(losses) / len(losses), 4)}")
    return net

# typically Tensor[64, 1024], Tensor[64, 1]
def massage(prompt: ImgPrompt) -> tuple[Tensor, Tensor]:
    text = prompt.embed #.reshape([512]).to(torch.float32).to(device)
    input_embed = torch.cat(
        [torch.cat([prompt.embed, cutout]).unsqueeze(0) for cutout in prompt.image_embed.to(device)]
    ).to(device)
    actual = Tensor([[prompt.label] for _ in prompt.image_embed]).to(device)
    return input_embed, actual


def validate(prompts: list[ImgPrompt], net: Optional[nn.Module] = None) -> None:
    if not net:
        net = torch.load("reaction_predictor.pth").to(device)  # type: ignore
    assert net
    loss_fn = nn.L1Loss()
    losses = []
    messages = []
    for i, prompt in enumerate(prompts):
        text = prompt.embed.reshape([512]).to(torch.float32)
        massaged = torch.cat(
            [torch.cat([text, cutout]).unsqueeze(0) for cutout in prompt.image_embed]
        ).to(device)
        prediction = net(massaged).to("cpu")
        actual = Tensor([prompt.label for _ in prompt.image_embed]).reshape(
            prediction.shape
        )
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

def main() -> None:
    prompts = torch.load("img_prompts.pth")  # type: ignore
    valid = len(prompts) // 5  # 20%
    train_set, valid_set = torch.utils.data.random_split(
        prompts, [len(prompts) - valid, valid]
    )
    print(len(train_set), len(valid_set))
    net = train(list(train_set))
    validate(list(valid_set), net)


def train_prod() -> None:
    prompts = torch.load("prompts.pth")  # type: ignore
    for prompt in prompts:
        prompt.embed = prompt.embed.to(device).to(torch.float32)
    train(prompts)


if __name__ == "__main__":
    main()
