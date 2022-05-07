import random

import torch
import tqdm
from torch import Tensor
from core import Prompt


def dist(embed1: Tensor, embed2: Tensor) -> float:
    return float(embed1.sub(embed2).norm(dim=2).div(2).arcsin().pow(2).mul(2))  # type: ignore


def knn(search_embed: Tensor, prompts: list[Prompt], k: int = 10) -> list[Prompt]:
    return sorted(prompts, key=lambda p: dist(p.embed, search_embed))[:k]


def weighted_knn(search_embed: Tensor, prompts: list[Prompt], k: int = 30) -> float:
    distances = sorted(
        [
            [distance, prompt.reacts]
            for prompt in prompts
            if (distance := float(dist(prompt.embed, search_embed)))
            != 0  # ignore the same point
        ]
    )
    total_weights = sum(1.0 / dist for dist, _ in distances[:k])
    return (
        sum([bool(reacts) * (1.0 / dist) for dist, reacts in distances[:k]])
        / total_weights
    )


def mse(data: list[tuple[float, float]]) -> float:
    return sum((pred - actual) ** 2 for pred, actual in data) / len(data)


def mae(data: list[tuple[float, float]]) -> float:
    return sum(abs(pred - actual) for pred, actual in data) / len(data)


def validate(prompts: list[Prompt]) -> None:
    search_space, search_keys = [], []
    for i, prompt in enumerate(prompts[:10000]):
        if i % 100 < 95:
            search_space.append(prompt)
        else:
            search_keys.append(prompt)
    random.shuffle(search_keys)
    k = 20
    data = [
        (weighted_knn(prompt.embed, search_space, k), prompt.label)
        for prompt in tqdm.tqdm(search_keys)
    ]
    # (sum(bool(p.reacts) for p in knn(prompt.embed, search_space, k)) / float(k)),
    print(mae(data))
    for i in range(10):
        print(f"predicted: {data[i][0]}, actual {data[i][1]}. {search_keys[i].prompt}")


# 100%|██████████████████████████████████████████████████████████████████████████████████| 500/500 [00:03<00:00, 136.18it/s]
# 100%|███████████████████████████████████████████████████████████████████████████████████| 100/100 [00:03<00:00, 25.16it/s]
# 0.5445
# predicted: 0.7, actual 0. technocratic thinking
# predicted: 0.45, actual 0. Julia made out of eggs
# predicted: 0.4, actual 1. space transportation using chiaroscuro
# predicted: 0.3, actual 1. derp
# predicted: 0.15, actual 0. a vigilante duckman prancing
# predicted: 0.15, actual 0. super Mario beating toadstool
# predicted: 0.65, actual 0. traffic
# predicted: 0.75, actual 0. Rapid eye movements in random access memories
# predicted: 0.45, actual 0. a functioning web 3 startup
# predicted: 0.4, actual 2. the wanderer cubist black and white with the words “wandering”


# notes:
# unbalanced: 0.20
# top20, weighted, balanced, bool: 0.255

# whole prompt set (validation likely in training set), MAE
# 0.46642516746930807

if __name__ == "__main__":
    validate(torch.load("prompts.pth"))  # type: ignore
