import asyncio
import base64
import logging
import os
from pathlib import Path
import aioredis
from aiohttp import web

app = web.Application()
redis = aioredis.from_url(os.environ["REDIS_URL"])

kube_cred = os.getenv("KUBE_CREDENTIALS")
if kube_cred:
    logging.info("kube creds")
    Path("/root/.kube").mkdir(exist_ok=True, parents=True)
    open("/root/.kube/config", "w").write(base64.b64decode(kube_cred).decode())

html = """
<!DOCTYPE HTML>
<iframe width="506" height="349"
    src="https://www.youtube.com/embed/live_stream?channel=UC_ViIZpK8aLEM_DKHO7W1YQ"
    frameborder="0" allowfullscreen="" style="float: left;">
</iframe>
<div style = "margin: 5%; float: right">
    prompts queue: </br>
    <ul>
        {prompts}
    </ul>
    <form method="post">
        add a prompt to the queue: <input name="prompt" value=""><br/>
        <input type="submit" value="submit">
    </form><br/>
    history:<br/>
    <ul>
        {history}
    </ul>
</div>
"""


async def get_output(cmd: str, inp: str = "") -> str:
    proc = await asyncio.create_subprocess_shell(cmd, stdin=-1, stdout=-1, stderr=-1)
    stdout, stderr = await proc.communicate(inp.encode())
    return stdout.decode().strip() or stderr.decode().strip()


async def index(request: web.Request) -> web.Response:
    if request.method == "POST":
        prompt = (await request.post()).get("prompt")
        if prompt:
            await redis.rpush("stream_queue", prompt)
    prompts = await redis.lrange("stream_queue", 0, -1)
    rendered_prompts = "\n".join([f"<li>{p.decode()}</li>" for p in prompts])
    history = await redis.lrange("stream_history", 0, -1)
    rendered_history = "\n".join([f"<li>{p.decode()}</li>" for p in history])
    return web.Response(
        body=html.format(prompts=rendered_prompts, history=rendered_history),
        content_type="text/html",
    )


async def mgmt(request: web.Request) -> web.Response:
    out = ""
    if request.method == "POST":
        action = (await request.post()).get("action")
        if action == "start":
            out = await get_output("kubectl apply -f stream.yaml")
        elif action == "stop":
            out = await get_output("kubectl delete pod stream")
    out = out.replace("\n", "<br/>\n")
    logs = (await get_output("kubectl logs --tail=100 stream")).replace("\n", "<br/>")
    body = f"""<!DOCTYPE HTML>
    <body style="font-family: monospace">
    <form method="post">
        <input type="submit" name="action" value="start"/><br/>
        <input type="submit" name="action" value="stop"/><br/>
    </form>
    <div style="float: right; margin 5%;">
        {out}<br/>
    </div>
    <div style="float: left; margin 5%;">
        {logs}<br/>
    </div>
    </body>
    """
    return web.Response(body=body, content_type="text/html")


app.add_routes([web.route("*", "/", index), web.route("*", "/mgmt", mgmt)])

if __name__ == "__main__":
    web.run_app(app, port=8080, host="0.0.0.0")
