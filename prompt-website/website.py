import os
import aioredis
from aiohttp import web

app = web.Application()
redis = aioredis.from_url(os.environ["REDIS_URL"])

html = """
<!DOCTYPE HTML>
<div style = "margin: 5%; float: left">
    prompts queue: </br>
    <ul>
        {prompts}
    </ul>
    <form method="post">
        add a prompt to the queue: <input name="prompt" value=""><br/>
        <input type="submit" value="submit">
    </form>
</div>
"""


async def index(request: web.Request) -> web.Response:
    if request.method == "POST":
        prompt = (await request.post()).get("prompt")
        if prompt:
            await redis.rpush("stream_queue", prompt)
    prompts = await redis.lrange("stream_queue", 0, -1)
    rendered_prompts = "\n".join([f"<li>{p.decode()}</li>" for p in prompts])
    return web.Response(
        body=html.format(prompts=rendered_prompts), content_type="text/html"
    )


app.add_routes([web.route("*", "/", index)])

if __name__ == "__main__":
    web.run_app(app, port=8080, host="0.0.0.0")
