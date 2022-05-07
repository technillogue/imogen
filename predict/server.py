import logging
import torch
from torch.nn import functional as F
from aiohttp import web

from clip import clip

logging.getLogger().setLevel("DEBUG")

device = "cpu"
net = torch.load("reaction_predictor", map_location=device)  # type: ignore
app = web.Application()

percep = clip.load("ViT-B/32", jit=False)[0]
percep.eval()


def norm(embedding: torch.Tensor) -> torch.Tensor:
    return F.normalize(embedding.unsqueeze(1), dim=2)


def embed(text: str) -> torch.Tensor:
    "normalized clip embedding of text"
    return norm(percep.encode_text(clip.tokenize(text, truncate=True)))


async def handle_request(request: web.Request) -> web.Response:
    text = await request.text()
    embedding = embed(text)
    score = float(net(embedding))
    return web.Response(status=200, text=str(score))


app.add_routes([web.get("/good", handle_request)])

if __name__ == "__main__":
    web.run_app(app, port=8080, host="0.0.0.0")
