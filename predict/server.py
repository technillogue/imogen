import logging
import torch
from aiohttp import web
from clip import clip
from core import embed
from knn import knn

prompts = torch.load("prompts.pth")

logging.basicConfig(
    format="{levelname} {module}:{lineno}: {message}", style="{", level="DEBUG"
)
app = web.Application()
device = "cpu"
net = torch.load("reaction_predictor.pth", map_location=device)  # type: ignore
perceptor = clip.load("ViT-B/32")[0]


async def handle_request(request: web.Request) -> web.Response:
    text = await request.text()
    embedding = embed(perceptor, text)
    tensor = net(embedding)
    logging.info(tensor)
    score = round(float(tensor), 4)
    logging.info(score)
    return web.Response(status=200, text=str(score))

async def handle_info(request: web.Request) -> web.Response:
    text = await request.text()
    embedding = embed(perceptor, text)
    score = round(float(net(embedding)), 4)
    closest = knn(embedding, prompts, k = 1)[0]
    return web.json_response({"score": str(score), "closest": closest.prompt, "reactions": closest.reacts})
    Response(status=200, text=str(score))



app.router.add_route("*", "/good", handle_request)
app.router.add_route("*", "/info", handle_info)

if __name__ == "__main__":
    web.run_app(app, port=8080, host="0.0.0.0")
