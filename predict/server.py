import logging
import torch
from aiohttp import web
from clip import clip
from core import embed

logging.basicConfig(
    format="{levelname} {module}:{lineno}: {message}", style="{", level="DEBUG"
)
app = web.Application()
device = "cpu"
net = torch.load("reaction_predictor", map_location=device)  # type: ignore
perceptor = clip.load("ViT-B/32")[0]
# perceptor.eval()


async def handle_request(request: web.Request) -> web.Response:
    text = await request.text()
    embedding = embed(perceptor, text)
    score = float(net(embedding))
    return web.Response(status=200, text=str(score))


app.router.add_route("*", "/good", handle_request)

if __name__ == "__main__":
    web.run_app(app, port=8080, host="0.0.0.0")
