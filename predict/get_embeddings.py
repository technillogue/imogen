import asyncio
from collections import defaultdict
import dataclasses
import os
import random
from typing import Callable
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
    "normalized clip embedding of text"
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


async def get_prompts() -> list[Prompt]:
    "just get every prompt"
    # n: int, skip: int = 0
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
    "pick the most reacted prompt from prompts with the same text"
    groups = defaultdict(list)
    for prompt in prompts:
        groups[prompt.prompt].append(prompt)
    bests: list[Prompt] = []
    for group in groups.values():
        bests.append(max(group, key=lambda prompt:prompt.reacts))
    return bests


def balance(
    prompts: list[Prompt], key: Callable = lambda prompt: bool(prompt.reacts)
) -> list[Prompt]:
    "shuffle together equal amounts of each group. default to equal amounts of prompts with and without reactions"
    groups = defaultdict(list)
    for prompt in prompts:
        groups[key(prompt)].append(prompt)
    cutoff = min(len(group) - 1 for group in groups.values())
    together = [prompt for group in groups.values() for prompt in group[:cutoff]]
    random.shuffle(together)
    return together


async def prepare() -> None:
    prompts = balance(pick_best(await get_prompts()))
    print(len(prompts))
    torch.save(prompts, "prompts.pth")


if __name__ == "__main__":
    asyncio.run(prepare())
