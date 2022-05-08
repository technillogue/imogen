import asyncio
import os
import random
from collections import defaultdict
from typing import Callable
import asyncpg
import torch
import tqdm
from clip import clip, model
from core import BasicPrompt, TokenPrompt, Prompt, embed


async def get_balanced_rows(n: int, skip: int = 0) -> list[BasicPrompt]:
    "get n class-balanced prompts, excluding id % 3 = skip"
    conn = await asyncpg.connect(os.getenv("DATABASE_URL") or os.getenv("dev_db"))
    # later: sample evenly from prompts with #n reactions
    query = """
    select prompt, map_len(reaction_map) as reacts, loss from prompt_queue
    where status='done' and group_id<>'' and id % 3 <> $1
    and map_len(reaction_map)::bool::int = $2
    order by random() limit $3
    """
    good = await conn.ftech(query, skip, 1, n // 2)
    bad = await conn.fetch(query, skip, 0, n // 2)
    await conn.close()
    return [BasicPrompt(**row) for row in good + bad]


async def get_all_rows() -> list[BasicPrompt]:
    "just get every prompt"
    conn = await asyncpg.connect(os.getenv("DATABASE_URL") or os.getenv("dev_db"))
    ret = await conn.fetch(
        """ select prompt, max(map_len(reaction_map)) as reacts,
        min(loss) as loss from prompt_queue
        where sent_ts is not null and status='done' and group_id<>''
        group by prompt;
        """
    )
    print(f"all rows: {len(ret)}")
    await conn.close()
    return [BasicPrompt(**row) for row in ret]


def pick_best(prompts: list[BasicPrompt]) -> list[BasicPrompt]:
    "pick the most reacted prompt from prompts with the same acceptable text"
    groups = defaultdict(list)
    # this group by is a bit redundant with the grouping in postgres so it's just filtering
    for prompt in prompts:
        if (
            any(c.isalpha() for c in prompt.prompt)
            and "/imagine" not in prompt.prompt
            and "in line" not in prompt.prompt
        ):
            groups[prompt.prompt].append(prompt)
    bests: list[BasicPrompt] = []
    for group in groups.values():
        bests.append(max(group, key=lambda prompt: prompt.reacts))
    return bests


def balance(
    prompts: list[BasicPrompt], key: Callable = lambda prompt: bool(prompt.reacts)
) -> list[BasicPrompt]:
    "shuffle together equal amounts of each group. default to equal amounts of prompts with and without reactions"
    groups = defaultdict(list)
    random.shuffle(prompts)  # make sure the within-group ordering is random
    for prompt in prompts:
        groups[key(prompt)].append(prompt)
    cutoff = min(len(group) - 1 for group in groups.values())
    together = [prompt for group in groups.values() for prompt in group[:cutoff]]
    random.shuffle(together)
    return together


def embed_all(perceptor: model.CLIP, prompts: list[BasicPrompt]) -> list[Prompt]:
    embedded_prompts: list[Prompt] = []
    for prompt in tqdm.tqdm(prompts):
        embedding = embed(perceptor, prompt.prompt).to("cpu").detach()
        embedded_prompts.append(
            Prompt(prompt.prompt, prompt.reacts, prompt.loss, embedding)
        )
    torch.cuda.empty_cache()
    return embedded_prompts


def tokenize_all(prompts: list[BasicPrompt]) -> list[TokenPrompt]:
    # cheating a bit
    return [
        TokenPrompt(
            prompt.prompt,
            prompt.reacts,
            prompt.loss,
            clip.tokenize(prompt.prompt, truncate=True),
        )
        for prompt in tqdm.tqdm(prompts)
    ]


async def prepare() -> None:
    all_basic_prompts = await get_all_rows()
    best = pick_best(all_basic_prompts)
    print(f"best: {len(best)}")
    kept_basic_prompts = balance(best)
    print(f"balanced: {len(kept_basic_prompts)}")
    perceptor = clip.load("ViT-B/32", jit=False)[0]
    perceptor.eval()
    prompts = embed_all(perceptor, kept_basic_prompts)
    print("embedded: ", len(prompts))
    torch.save(prompts, "prompts.pth")


async def prepare_basic() -> None:
    torch.save(balance(pick_best(await get_all_rows())), "basic_prompts.pth")


async def prepare_token() -> None:
    torch.save(
        tokenize_all(balance(pick_best(await get_all_rows()))), "token_prompts.pth"
    )


if __name__ == "__main__":
    asyncio.run(prepare_token())
