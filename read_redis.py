#!/usr/bin/python3.9
# pylint: disable=consider-using-with,subprocess-run-check
import json
import logging
import os
import subprocess
import sys
import time
import traceback

import redis
import requests
import TwitterAPI as t

import main_ganclip_hacking as clipart

logging.getLogger().setLevel("DEBUG")
twitter_api = t.TwitterAPI(
    "qxmCL5ebziSwOIlf3MByuhRvY",
    "3sj1HeUXPeZ3YEG45j1fa1ckGvCQI2lTmg39QUue1bK69KPtGL",
    "1442633760315375621-UreMIwMZK3x7Povds8A4ruEbS7VPeD",
    "INQ5JoET33lxjoIyT8VO557iPFd9Y2uAuxhZbUUeepzQq",
    api_version="1.1",
)
username = "@dreambs3"
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


def post(elapsed: float, prompt_blob: dict, loss: str, fname="progress.jpg") -> None:
    minutes, seconds = divmod(elapsed, 60)
    f = open(fname, mode="rb")
    message = f"{prompt_blob['prompt']}\nTook {minutes}m{seconds}s to generate, {loss} loss, v{clipart.version}."
    requests.post(
        f"{signal_url}/attachment",
        params={"message": message, "destination": prompt_blob["callback"]},
        files={"image": f},
    )
    media = twitter_api.request(
        "media/upload", None, {"media": open(fname, mode="rb").read()}
    ).json()
    media_id = media["media_id"]
    twitter_post = {
        "status": prompt_blob["prompt"],
        "media_ids": media_id,
    }
    twitter_api.request("statuses/update", twitter_post)


def admin(msg: str) -> None:
    requests.post(
        f"{signal_url}/admin",
        params={"message": msg},
    )


if __name__ == "__main__":
    while 1:
        item = r.lindex("prompt_queue", 0)
        print(item)
        if not item:
            time.sleep(60)
            item = r.lindex("prompt_queue", 0)
            if not item:
                admin("powering down worker")
                subprocess.run(["sudo", "poweroff"])
                continue
        try:
            blob = json.loads(item)
        except (json.JSONDecodeError, TypeError):
            logging.info(item)
            continue
        try:
            settings = json.loads(blob["prompt"])
            assert isinstance(settings, dict)
            args = clipart.base_args.with_update({"max_iterations": 200}).with_update(
                settings
            )
        except (json.JSONDecodeError, AssertionError):
            args = clipart.base_args.with_update(
                {"text": blob["prompt"], "max_iterations": 200}
            )
        print(args)
        start_time = time.time()
        try:
            loss = clipart.generate(args)
            print("generated")
            post(round(time.time() - start_time), blob, loss)
            r.lrem("prompt_queue", 1, item)
        except:  # pylint: disable=bare-except
            error_message = traceback.format_exc()
            print(item)
            admin(item)
            print(error_message)
            admin(error_message)
