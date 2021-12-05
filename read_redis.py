#!/usr/bin/python3.9
# pylint: disable=consider-using-with,subprocess-run-check
import json
import logging
import os
import subprocess
import sys
import time
import traceback
import cProfile
import redis
import requests
import TwitterAPI as t
import mk_video

# import main_ganclip_hacking as clipart
import better_imagegen as clipart

logging.getLogger().setLevel("DEBUG")
twitter_api = t.TwitterAPI(
    "qxmCL5ebziSwOIlf3MByuhRvY",
    "3sj1HeUXPeZ3YEG45j1fa1ckGvCQI2lTmg39QUue1bK69KPtGL",
    "1442633760315375621-UreMIwMZK3x7Povds8A4ruEbS7VPeD",
    "INQ5JoET33lxjoIyT8VO557iPFd9Y2uAuxhZbUUeepzQq",
    api_version="1.1",
)
username = "@dreambs3"
handler = logging.FileHandler("/home/ubuntu/info.log")
handler.setLevel("INFO")
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
    url = "redis://:speak-friend-and-enter@forest-redis.fly.dev:10000"
    # "redis://:Ing2ODJrcXA2bXZqOWQ1NDMia953abbc82e1f4b9c47158d739526833d1006263@imogen.redis.fly.io:10079"
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
    media_resp = twitter_api.request(
        "media/upload", None, {"media": open(fname, mode="rb").read()}
    )
    try:
        media = media_resp.json()
        media_id = media["media_id"]
        twitter_post = {
            "status": prompt_blob["prompt"],
            "media_ids": media_id,
        }
        twitter_api.request("statuses/update", twitter_post)
    except KeyError:
        try:
            logging.error(media_resp.text)
            admin(media_resp.text)
        except: # pylint: disable=bare-except
            pass


def admin(msg: str) -> None:
    requests.post(
        f"{signal_url}/admin",
        params={"message": str(msg)},
    )


if __name__ == "__main__":
    backoff = 60
    while 1:
        try:
            item = r.lindex("prompt_queue", 0)
            print(item)
            if not item:
                time.sleep(60)
                item = r.lindex("prompt_queue", 0)
                if not item:
                    admin("powering down worker")
                    if not os.getenv("NO_POWEROFF"):
                        subprocess.run(["sudo", "poweroff"])
                    continue
            try:
                blob = json.loads(item)
            except (json.JSONDecodeError, TypeError):
                logging.info(item)
                continue
            video = False
            try:
                settings = json.loads(blob["prompt"])
                assert isinstance(settings, dict)
                args = clipart.base_args.with_update(
                    {"max_iterations": 200}
                ).with_update(settings)
                video = settings.get("video", False)
            except (json.JSONDecodeError, AssertionError):
                maybe_prompt_list = [p.strip() for p in blob["prompt"].split("//")]
                video = len(maybe_prompt_list) > 1
                if video:
                    args = clipart.base_args.with_update({"prompts": maybe_prompt_list})
                else:
                    args = clipart.base_args.with_update(
                        {"text": blob["prompt"], "max_iterations": 200}
                    )
            params = blob.get("params", {})
            if params.get("init_image"):
                open(params["init_image"], "wb").write(r.get(params["init_image"]))
            args = args.with_update(blob.get("params", {}))
            path = f"output/{clipart.mk_slug(args.prompts)}"
            print(args)
            start_time = time.time()
            if args.profile:
                with cProfile.Profile() as profiler:
                    loss = clipart.generate(args)
                profiler.dump_stats(f"profiling/{clipart.version}.stats")
                print("generated with stats")
            else:
                loss = clipart.generate(args)
                print("generated")
                if video:
                    mk_video.video(path)
            fname = "video.mp4" if video else "progress.png"
            post(round(time.time() - start_time), blob, loss, f"{path}/{fname}")
            r.lrem("prompt_queue", 1, item)
            backoff = 60
        except:  # pylint: disable=bare-except
            error_message = traceback.format_exc()
            print(item)
            admin(item)
            print(error_message)
            admin(error_message)
            time.sleep(backoff)
            backoff *= 1.5
