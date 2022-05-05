import asyncio
import dataclasses
import math
import os
import sys
import asyncpg
import clip
import torch
import tqdm
from torch import Tensor
from torch.nn import functional as F

device = "cuda:0"  # "cpu"
percep = clip.load("ViT-B/32", jit=False)[0].to(device)
percep.eval()


def norm(embedding: Tensor) -> Tensor:
    return F.normalize(embedding.unsqueeze(1), dim=2)


def embed(text: str) -> Tensor:
    return norm(percep.encode_text(clip.tokenize(text, truncate=True).to(device)))


def dist(embed1: Tensor, embed2: Tensor) -> Tensor:
    return embed1.sub(embed2).norm(dim=2).div(2).arcsin().pow(2).mul(2)


@dataclasses.dataclass
class Prompt:
    prompt: str
    reacts: int
    loss: float
    embed: Tensor


async def get_prompts(n: int) -> list[Prompt]:
    conn = await asyncpg.connect(os.getenv("DATABASE_URL") or os.getenv("dev_db"))
    prompts = []
    # later: sample evenly from prompts with #n reactions
    ret = await conn.fetch(
        "select prompt, map_len(reaction_map) as reacts, loss from prompt_queue where status='done' limit $1",
        n,
    )
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
    await conn.close()
    return prompts


def knn(search_text: str, prompts: list[Prompt], k: int = 10) -> list[Prompt]:
    search_embed = embed(search_text)
    return sorted(prompts, key=lambda p: float(dist(p.embed, search_embed)))[:k]


def mse(data: list[tuple[float, float]]) -> float:
    return sum((pred - actual)**2 for pred, actual in data) / len(data)


async def validate() -> None:
    train_lol = await get_prompts(500)
    conn = await asyncpg.connect(os.getenv("DATABASE_URL") or os.getenv("dev_db"))
    test = await conn.fetch(
        "select prompt, map_len(reaction_map) as reacts, loss from prompt_queue where status='done' order by random() limit 100"
    )
    await conn.close()
    data = []
    for row in tqdm.tqdm(test):
        data.append(
            (
                (sum(p.reacts for p in knn(row["prompt"], train_lol, 20)) / 20),
                row["reacts"],
            )
        )
    print(mse(data))
    for i in range(10):
        print(f"predicted: {data[i][0]}, actual {data[i][1]}. {test[i]['prompt']}")
    import pdb

    pdb.set_trace()


async def main(text: str) -> None:
    prompts = await get_prompts(1000)
    torch.save(prompts, "prompts.pth")
    results = knn(text or " ".join(sys.argv[1:]), prompts)
    for result in results:
        print(f"{result.prompt}: {result.reacts}")
    print("average: ", sum(r.reacts for r in results) / len(results))


if __name__ == "__main__":
    asyncio.run(validate())
