import asyncio
from collections import defaultdict
import dataclasses
import os
import random
import sys
from typing import Any, Callable, cast
import asyncpg
import torch
import tqdm
from clip import clip
from torch import Tensor
from torch.nn import functional as F

device = "cpu"
percep = clip.load("ViT-B/32", jit=False)[0].to(device)
percep.eval()


def norm(embedding: Tensor) -> Tensor:
    return F.normalize(embedding.unsqueeze(1), dim=2)


def embed(text: str) -> Tensor:
    return (
        norm(percep.encode_text(clip.tokenize(text, truncate=True).to(device)))
        .to("cpu")
        .detach()
    )


@dataclasses.dataclass
class Prompt:
    prompt: str
    reacts: int
    loss: float
    embed: Tensor


async def get_prompts(n: int, skip: int = 0) -> list[Prompt]:
    conn = await asyncpg.connect(os.getenv("DATABASE_URL") or os.getenv("dev_db"))
    prompts = []
    # later: sample evenly from prompts with #n reactions
    # ret = (
    #     await conn.fetch(
    #         """select prompt, map_len(reaction_map) as reacts, loss from prompt_queue
    #     where status='done' and group_id<>'' and id % 3 <> $2 and map_len(reaction_map) = 0 order by random() limit $1 """,
    #         n // 2, skip
    #     )
    #     + await conn.fetch(
    #         """select prompt, map_len(reaction_map) as reacts, loss from prompt_queue
    #     where status='done' and group_id<>'' and id % 3 <> $2 and map_len(reaction_map)<>0 order by random() limit $1 """,
    #         n // 2, skip
    #     )
    # )
    ret = await conn.fetch(
        """select prompt, map_len(reaction_map) as reacts, loss from prompt_queue
        where status='done' and group_id<>'' """
    )
    await conn.close()
    for record in tqdm.tqdm(ret):
        try:
            prompts.append(
                Prompt(
                    record["prompt"],
                    record["reacts"],
                    record["loss"],
                    embed(record["prompt"]),
                )
            )
        except RuntimeError:
            break
    torch.cuda.empty_cache()
    return prompts


def pick_best(prompts: list[Prompt]) -> list[Prompt]:
    groups = defaultdict(list)
    for prompt in prompts:
        groups[prompt.prompt].append(prompt)
    bests: list[Prompt] = []
    for group in groups.values():
        group[0].reacts = max(prompt.reacts for prompt in group)
        bests.append(group[0])
    return bests


def balance(
    prompts: list[Prompt], key: Callable = lambda prompt: bool(prompt.reacts)
) -> list[Prompt]:
    groups = defaultdict(list)
    for prompt in prompts:
        groups[key(prompt)].append(prompt)
    cutoff = min(len(group) - 1 for group in groups.values())
    together = [prompt for group in groups.values() for prompt in group[:cutoff]]
    random.shuffle(together)
    return together


def dist(embed1: Tensor, embed2: Tensor) -> float:
    return float(embed1.sub(embed2).norm(dim=2).div(2).arcsin().pow(2).mul(2))  # type: ignore


def knn(search_text: str, prompts: list[Prompt], k: int = 10) -> list[Prompt]:
    search_embed = embed(search_text)
    return sorted(prompts, key=lambda p: float(dist(p.embed, search_embed)))[:k]


def weighted_knn(
    search_text: str, prompts: list[Prompt], k: int = 30, fn: Callable = bool
) -> Any:
    search_embed = embed(search_text)
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
        sum([fn(reacts) * (1.0 / dist) for dist, reacts in distances[:k]])
        / total_weights
    )


def mse(data: list[tuple[float, float]]) -> float:
    return sum((pred - actual) ** 2 for pred, actual in data) / len(data)


async def validate() -> None:
    try:
        untyped = torch.load("prompts.pth")  # type: ignore
        train_lol = cast(list[Prompt], untyped)
    except FileNotFoundError:
        train_lol = await get_prompts(600)
        torch.save(train_lol, "prompts.pth")
    conn = await asyncpg.connect(os.getenv("DATABASE_URL") or os.getenv("dev_db"))
    test = (
        await conn.fetch(
            """select prompt, map_len(reaction_map) as reacts, loss from prompt_queue
        where status='done' and group_id<>'' and id % 3 = 0 and map_len(reaction_map) = 0 order by random() limit 50"""
        )
        + await conn.fetch(
            """select prompt, map_len(reaction_map) as reacts, loss from prompt_queue
        where status='done' and group_id<>'' and id % 3 = 0
        and map_len(reaction_map) <> 0 order by random() limit 50"""
        )
    )
    random.shuffle(test)
    await conn.close()
    data = []
    k = 20
    for row in tqdm.tqdm(test):
        data.append(
            (
                weighted_knn(row["prompt"], train_lol, k),
                # (sum(bool(p.reacts) for p in knn(row["prompt"], train_lol, k)) / float(k)),
                float(bool(row["reacts"])),
            )
        )
    print(mse(data))
    for i in range(10):
        print(f"predicted: {data[i][0]}, actual {data[i][1]}. {test[i]['prompt']}")
    import pdb

    pdb.set_trace()


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


async def main(text: str) -> None:
    prompts = await get_prompts(1000)
    torch.save(prompts, "prompts.pth")
    results = knn(text or " ".join(sys.argv[1:]), prompts)
    for result in results:
        print(f"{result.prompt}: {result.reacts}")
    print("average: ", sum(r.reacts for r in results) / len(results))


async def prepare() -> None:
    prompts = balance(pick_best(await get_prompts(-1)))
    print(len(prompts))
    torch.save(prompts, "prompts.pth")


if __name__ == "__main__":
    asyncio.run(prepare())
