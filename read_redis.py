#!/usr/bin/python3.9
import logging
import time
import json
import sys
import subprocess
import os
import redis
import requests
import main_ganclip_hacking as clipart

handler = logging.FileHandler("/home/ubuntu/debug.log")
handler.setLevel("DEBUG")
logging.getLogger().addHandler(handler)
logging.info("starting")
logging.debug("debug")
tee = subprocess.Popen(["tee", "-a", "fulllog.txt"], stdin=subprocess.PIPE)
# Cause tee's stdin to get a copy of our stdin/stdout (as well as that
# of any child processes we spawn)
os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

signal_url = "https://imogen.fly.dev"
# old_url = "redis://:ImVqcG9uMTdqMjc2MWRncjQi8a6c817565c7926c7c7e971b4782cf96a705bb20@forest-dev.redis.fly.io:10079"
requests.post(f"{signal_url}/admin", params={"message": "starting read_redis"})
try:
    url = sys.argv[1]
except IndexError:
    url = "redis://:Ing2ODJrcXA2bXZqOWQ1NDMia953abbc82e1f4b9c47158d739526833d1006263@imogen.redis.fly.io:10079"
password, rest = url.removeprefix("redis://:").split("@")
host, port = rest.split(":")
r = redis.Redis(host=host, port=port, password=password)
while 1:
    item = r.lindex("prompt_queue", 0)
    if not item:
        time.sleep(60)
        item = r.lindex("prompt_queue", 0)
        if not item:
            requests.post(
                f"{signal_url}/admin", params={"message": "powering down worker"}
            )
            #subprocess.run(["sudo", "poweroff"])
    try:
        blob = json.loads(item)
    except json.JSONDecodeError:
        logging.info(item)
        continue
    try:
        settings = json.loads(blob["prompt"])
        args = clipart.base_args.with_update(settings)
    except json.JSONDecodeError:
        args = clipart.base_args.with_update(
            {"text": blob["prompt"], "max_iterations": 100}
        )
    print(args)
    start_time = time.time()
    clipart.generate(args)
    minutes, seconds = divmod(round(time.time() - start_time), 60)
    f = open("progress.jpg", mode="rb")
    print("generated")
    message = blob["prompt"] + f".\nTook {minutes}m{seconds}s to generate"
    requests.post(
        f"{signal_url}/attachment",
        params={"message": message, "destination": blob["callback"]},
        files={"image": f},
    )
    r.lrem("prompt_queue", 1, item)
