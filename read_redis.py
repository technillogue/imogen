import time
import json
import subprocess

import redis
import requests
import main_ganclip_hacking as clipart

requests.post("https://imogen.fly.dev/admin", params={"message": "starting read_redis"})
url = "redis://:ImVqcG9uMTc2OHA2cWRncjQie92871a0d1480b7ebe533d1861ed27b7ca8590bc@imogen.redis.fly.io:10079"

password, rest = url.removeprefix("redis://:").split("@")
host, port = rest.split(":")
r = redis.Redis(host=host, port=port, password=password)
while 1:
    item = r.lindex("prompt_queue", 0)
    if item is None:
        time.sleep(60)
        item = r.lindex("prompt_queue", 0)
        if not item:
            requests.post("https://imogen.fly.dev/admin", params={"message": "powering down worker"})
            subprocess.run(["sudo", "poweroff"])
    blob = json.loads(item)
    args = clipart.base_args.with_update({"text": blob["prompt"], "max_iterations": 50})
    print(args)
    clipart.generate(args)
    f = open("progress.png", mode="rb").read()
    requests.post(f"https://imogen.fly.dev/attachment/{blob['callback']}", params={"message": blob["prompt"]}, files={"image": f})
    r.lrem("prompt_queue", item, 1)
