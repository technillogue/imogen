#!/usr/bin/python3.9
# pylint: disable=consider-using-with,subprocess-run-check
import cProfile
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from typing import Optional

import psycopg
import requests
import TwitterAPI as t

import better_imagegen as clipart
import feedforward
import mk_video

logging.getLogger().setLevel("DEBUG")
twitter_api = t.TwitterAPI(
    "qxmCL5ebziSwOIlf3MByuhRvY",
    "3sj1HeUXPeZ3YEG45j1fa1ckGvCQI2lTmg39QUue1bK69KPtGL",
    "1442633760315375621-UreMIwMZK3x7Povds8A4ruEbS7VPeD",
    "INQ5JoET33lxjoIyT8VO557iPFd9Y2uAuxhZbUUeepzQq",
    api_version="1.1",
)
username = "@dreambs3"
handler = logging.FileHandler("info.log")
handler.setLevel("INFO")
logging.getLogger().addHandler(handler)
logging.info("starting")
logging.debug("debug")
tee = subprocess.Popen(["tee", "-a", "fulllog.txt"], stdin=subprocess.PIPE)
# Cause tee's stdin to get a copy of our stdin/stdout (as well as that
# of any child processes we spawn)
os.dup2(tee.stdin.fileno(), sys.stdout.fileno())  # type: ignore
os.dup2(tee.stdin.fileno(), sys.stderr.fileno())  # type: ignore

signal_url = "https://imogen-renaissance.fly.dev"
requests.post(f"{signal_url}/admin", params={"message": "starting read_redis"})
try:
    url = sys.argv[1]
except IndexError:
    url = "redis://:speak-friend-and-enter@forest-redis.fly.dev:10000"
    # "redis://:Ing2ODJrcXA2bXZqOWQ1NDMia953abbc82e1f4b9c47158d739526833d1006263@imogen.redis.fly.io:10079"
password, rest = url.removeprefix("redis://:").split("@")
host, port = rest.split(":")
r = redis.Redis(host=host, port=int(port), password=password)


def post(
    elapsed: float,
    prompt_blob: dict,
    fname: str = "progress.jpg",
    loss: Optional[float] = None,
) -> None:
    minutes, seconds = divmod(elapsed, 60)
    f = open(fname, mode="rb")
    message = f"{prompt_blob['prompt']}\nTook {minutes}m{seconds}s to generate,"
    if loss:
        message += f"{loss} loss,"
    message += f" v{clipart.version}."
    url = prompt_blob.get("url", signal_url)
    requests.post(
        f"{url}/attachment",
        params={
            "message": message,
            "destination": prompt_blob["callback"],
            "author": prompt_blob.get("author", ""),
            "timestamp": prompt_blob.get("timestamp", ""),
        },
        files={"image": f},
    )
    if not fname.endswith("mp4"):
        media_resp = twitter_api.request(
            "media/upload", None, {"media": open(fname, mode="rb").read()}
        )
    else:
        bytes_sent = 0
        total_bytes = os.path.getsize(fname)
        file = open(fname, "rb")
        r = twitter_api.request(
            "media/upload",
            {"command": "INIT", "media_type": "video/mp4", "total_bytes": total_bytes},
        )

        media_id = r.json()["media_id"]
        segment_id = 0

        while bytes_sent < total_bytes:
            chunk = file.read(4 * 1024 * 1024)
            r = twitter_api.request(
                "media/upload",
                {
                    "command": "APPEND",
                    "media_id": media_id,
                    "segment_index": segment_id,
                },
                {"media": chunk},
            )
            segment_id = segment_id + 1
            bytes_sent = file.tell()
            logging.debug("[" + str(total_bytes) + "]", str(bytes_sent))

        media_resp = twitter_api.request(
            "media/upload", {"command": "FINALIZE", "media_id": media_id}
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
        except:  # pylint: disable=bare-except
            pass


def admin(msg: str) -> None:
    requests.post(
        f"{signal_url}/admin",
        params={"message": str(msg)},
    )


def handle_item(item: bytes) -> None:
    try:
        blob = json.loads(item)
    except (json.JSONDecodeError, TypeError):
        logging.info(item)
        return
    video = False
    try:
        settings = json.loads(blob["prompt"])
        assert isinstance(settings, dict)
        args = clipart.base_args.with_update({"max_iterations": 220}).with_update(
            settings
        )
        video = settings.get("video", False)
    except (json.JSONDecodeError, AssertionError):
        maybe_prompt_list = [p.strip() for p in blob["prompt"].split("//")]
        video = len(maybe_prompt_list) > 1
        if video:
            args = clipart.base_args.with_update(
                {"prompts": maybe_prompt_list, "max_iterations": 1000}
            )
        else:
            args = clipart.base_args.with_update(
                {"text": blob["prompt"], "max_iterations": 200}
            )
    params = blob.get("params", {})
    if params.get("init_image"):
        open(params["init_image"], "wb").write(r[params["init_image"]])
    params["video"] = video
    args = args.with_update(blob.get("params", {}))
    path = f"output/{clipart.mk_slug(args.prompts)}"
    print(args)
    start_time = time.time()
    if blob.get("feedforward"):
        feedforward_path = (
            f"results/single/{feedforward.mk_slug(blob['prompt'])}/progress.png"
        )
        loss = feedforward.generate(blob)
        post(round(time.time() - start_time), blob, feedforward_path, round(loss, 4))
        return
    if blob.get("feedforward_fast"):
        feedforward_path = f"results/single/{feedforward.mk_slug(blob['prompt'])}.png"
        loss = feedforward.generate_forward(blob, out_path=feedforward_path)
        post(round(time.time() - start_time), blob, feedforward_path)
        return
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
    post(round(time.time() - start_time), blob, f"{path}/{fname}", round(loss, 4))
    return


if __name__ == "__main__":
    backoff = 60.0
    with psycopg.connect(os.getenv("DATABASE_URL")) as conn:
        while 1:
            conn.execute(
                """UPDATE prompt_queue SET status='pending', assigned_at=null
                WHERE status='assigned' AND assigned_at  < (now() - interval '10 minutes');"""
            )
            paid = (
                "AND paid=TRUE" or ""
            )  # should be used by every instance except the first
            prompt_id = conn.execute(
                f"SELECT id FROM prompt_queue WHERE status='pending' {paid} ORDER BY ts ASC, paid DESC LIMIT 1;"
            ).fetchone()[0]
            if not prompt_id:
                pass  # sleep? poweroff?
            prompt, params, url = conn.execute(
                "UPDATE prompt_queue SET status='assigned', assigned_at=now() WHERE ts = $1 RETURNING prompt, params, url;",
                prompt_id
            ).fetchone()
            handle_item(prompt, parms, url)
            conn.execute("UPDATE prompt_queue SET status='done' WHERE ts=$1;", prompt_id)


    while 1:
        try:
            item = r.lindex("prompt_queue", 0)  # type: bytes
            print(item)
            if not item:
                time.sleep(60)
                item = r.lindex("prompt_queue", 0)
                if not item:
                    if os.getenv("POWEROFF"):
                        admin("powering down worker")
                        subprocess.run(["sudo", "poweroff"])
                    continue
            handle_item(item)
            r.lrem("prompt_queue", 1, item)
            # r.rpoplpush("prompt_queue", "processing-{host}") # ttl?
            # r.lrem("processing-{host}"
            backoff = 60
        except redis.exceptions.ConnectionError:
            continue
        except Exception as e:  # pylint: disable=bare-except
            error_message = traceback.format_exc()
            if item:
                print(item)
                admin(item)
            print(error_message)
            admin(error_message)
            if "out of memory" in str(e):
                sys.exit(137)
            time.sleep(backoff)
            backoff *= 1.5
