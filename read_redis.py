import time
import json
import subprocess

import redis
import requests
import main_ganclip_hacking as clipart

signal_url = "https://fast-vampirebat-51.loca.lt"
#old_url = "redis://:ImVqcG9uMTdqMjc2MWRncjQi8a6c817565c7926c7c7e971b4782cf96a705bb20@forest-dev.redis.fly.io:10079"
#redis_imogen_url = "redis://:ImVqcG9uMTc2OHA2cWRncjQie92871a0d1480b7ebe533d1861ed27b7ca8590bc@imogen.redis.fly.io:10079"
requests.post(f"{signal_url}/admin", params={"message": "starting read_redis"})
url = "redis://:ImVqcG9uMTdqMjc2MWRncjQi8a6c817565c7926c7c7e971b4782cf96a705bb20@forest-dev.redis.fly.io:10079"
>>>>>>> 2821fa4c097abb73255f156f0887f3e6c5de0950
password, rest = url.removeprefix("redis://:").split("@")
host, port = rest.split(":")
r = redis.Redis(host=host, port=port, password=password)
while 1:
    item = r.lindex("prompt_queue", 0)
    if item is None:
        time.sleep(60)
        item = r.lindex("prompt_queue", 0)
        if not item:
            requests.post(f"{signal_url}/admin", params={"message": "powering down worker"})
            subprocess.run(["sudo", "poweroff"])
    blob = json.loads(item)
    args = clipart.base_args.with_update({"text": blob["prompt"], "max_iterations": 50})
    print(args)
    clipart.generate(args)
    f = open("progress.png", mode="rb").read()
    requests.post(f"{signal_url}/attachment/{blob['callback']}", params={"message": blob["prompt"]}, files={"image": f})
    r.lrem("prompt_queue", item, 1)
