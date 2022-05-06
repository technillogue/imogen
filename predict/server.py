import torch
from get_embeddings import embed
from aiohttp import web

net = torch.load("reaction_predictor.pth")  # type: ignore
app = web.Application()


async def handle_request(request: web.Request) -> web.Response:
    text = await request.text()
    embedding = embed(text)
    score = float(net(embedding))
    return web.Response(status=200, text=str(score))


app.add_routes([web.get("/good", handle_request)])

if __name__ == "__main__":
    web.run_app(app, port=8080, host="0.0.0.0")
