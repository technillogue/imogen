#!/usr/bin/python3.9
# pylint: disable=subprocess-run-check
import cProfile
import dataclasses
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from typing import Optional

import psycopg
import redis
import requests
import TwitterAPI as t
from psycopg.rows import class_row

import better_imagegen as clipart
import feedforward
import mk_video
import utils

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

admin_signal_url = "https://imogen-renaissance.fly.dev"

try:
    url = sys.argv[1]
except IndexError:
    url = "redis://:speak-friend-and-enter@forest-redis.fly.dev:10000"
    # "redis://:Ing2ODJrcXA2bXZqOWQ1NDMia953abbc82e1f4b9c47158d739526833d1006263@imogen.redis.fly.io:10079"
password, rest = url.removeprefix("redis://:").split("@")
host, port = rest.split(":")
r = redis.Redis(host=host, port=int(port), password=password)


def admin(msg: str) -> None:
    requests.post(
        f"{admin_signal_url}/admin",
        params={"message": str(msg)},
    )


def stop() -> None:
    if os.getenv("POWEROFF"):
        admin("powering down worker")
        subprocess.run(["sudo", "poweroff"])
    else:
        time.sleep(60)


@dataclasses.dataclass
class Prompt:
    prompt_id: int
    prompt: str
    url: str
    slug: str = ""
    params: str = ""
    param_dict: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            self.param_dict = json.loads(self.params)
            assert isinstance(self.param_dict, dict)
        except (json.JSONDecodeError, AssertionError):
            self.param_dict = {}
        self.slug = clipart.mk_slug(self.prompt)


@dataclasses.dataclass
class Result:
    elapsed: int
    loss: float
    filepath: str


def get_prompt(conn: psycopg.Connection) -> Optional[Prompt]:
    conn.execute(
        """UPDATE prompt_queue SET status='pending', assigned_at=null
        WHERE status='assigned' AND assigned_at  < (now() - interval '10 minutes');"""
    )  # maybe this is a trigger
    paid = ""  # "AND paid=TRUE" # should be used by every instance except the first
    maybe_id = conn.execute(
        f"SELECT id FROM prompt_queue WHERE status='pending' {paid} ORDER BY signal_ts ASC LIMIT 1;"
    ).fetchone()
    if not maybe_id:
        return None
    prompt_id = maybe_id[0]
    maybe_prompt = conn.execute(
        "UPDATE prompt_queue SET status='assigned', assigned_at=now() WHERE id = $1 RETURNING id, prompt, params, url;",
        prompt_id,
        row_factory=class_row(Prompt),
    ).fetchone()
    return maybe_prompt


def main() -> None:
    admin("starting read_redis")
    # clear failed instances
    # try to get an id. if we can't, there's no work, and we should stop
    # try to claim it. if we can't, someone else took it, and we should try again
    # generate the prompt
    backoff = 60.0
    # catch some database connection errors
    with psycopg.connect(utils.get_secret("DATABASE_URL")) as conn:
        while 1:
            # try to claim
            prompt = get_prompt(conn)
            if not prompt:
                stop()
                continue
            print(prompt)
            try:
                result = handle_item(prompt)
                # success
                fmt = """UPDATE prompt_queue SET status='uploading', loss=$1, elapsed_gpu=$2, filename=$3, WHERE id=$4;"""
                conn.execute(
                    fmt,
                    result.loss,
                    result.elapsed,
                    result.filepath,
                    prompt.prompt_id,
                )
                post(result, prompt)
                conn.execute(
                    "UPDATE prompt_queue SET status='done' WHERE id=$1",
                    prompt.prompt_id,
                )
                backoff = 60
            except Exception as e:  # pylint: disable=broad-except
                error_message = traceback.format_exc()
                if prompt:
                    print(prompt)
                    admin(repr(prompt))
                print(error_message)
                admin(error_message)
                if "out of memory" in str(e):
                    sys.exit(137)
                time.sleep(backoff)
                backoff *= 1.5


# parse raw parameters
# parse prompt list
# it's either a specific function or the default one
# for imagegen, if there's an initial image, download it from postgres or redis
# pick a slug
# pass maybe raw parameters and initial parameters to the function to get loss and a file
# if it's a list of prompts, generate a video using the slug
# (ideally the function takes care of this though and writes directly to ffmpeg)
# at this point ideally we need to mark that we generated it, but it wasn't sent yet.
# (maybe move it to goog's s3)
# make a message with the prompt, time, loss, and version
# upload the file, id, and message to imogen based on the url. ideally retry on non-200
# (imogen looks up destination, author, timestamp to send).
# upload to twitter. if it fails, maybe log video size


def handle_item(prompt: Prompt) -> Result:
    video = False
    try:
        settings = json.loads(prompt.prompt)
        assert isinstance(settings, dict)
        args = clipart.base_args.with_update({"max_iterations": 221}).with_update(
            settings
        )
        video = settings.get("video", False)
    except (json.JSONDecodeError, AssertionError):
        maybe_prompt_list = [p.strip() for p in prompt.prompt.split("//")]
        video = len(maybe_prompt_list) > 1
        if video:
            args = clipart.base_args.with_update(
                {"prompts": maybe_prompt_list, "max_iterations": 1000}
            )
        else:
            args = clipart.base_args.with_update(
                {"text": prompt.prompt, "max_iterations": 221}
            )
    if prompt.param_dict.get("init_image"):
        # download the image from redis
        open(prompt.param_dict["init_image"], "wb").write(
            r[prompt.param_dict["init_image"]]
        )
    prompt.param_dict["video"] = video
    args = args.with_update(prompt.param_dict)
    print(args)
    path = f"output/{clipart.mk_slug(args.prompts)}"
    feedforward_path = ""
    start_time = time.time()
    if prompt.param_dict.get("feedforward"):
        feedforward_path = f"results/single/{prompt.slug}/progress.png"
        loss = feedforward.generate(prompt.prompt)
    elif prompt.param_dict.get("feedforward_fast"):
        feedforward_path = f"results/single/{prompt.slug}.png"
        loss = feedforward.generate_forward(prompt.prompt, out_path=feedforward_path)
    elif args.profile:
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
    return Result(
        elapsed=round(time.time() - start_time),
        filepath=feedforward_path or f"{path}/{fname}",
        loss=round(loss, 4),
    )


def post(result: Result, prompt: Prompt) -> None:
    minutes, seconds = divmod(result.elapsed, 60)
    f = open(result.filepath, mode="rb")
    message = f"{prompt.prompt}\nTook {minutes}m{seconds}s to generate,"
    if result.loss:
        message += f"{result.loss} loss,"
    message += f" v{clipart.version}."
    requests.post(
        f"{prompt.url or admin_signal_url}/attachment",
        params={"message": message, "id": str(prompt.prompt_id)},
        files={"image": f},
    )
    post_tweet(result, prompt)


def post_tweet(result: Result, prompt: Prompt) -> None:
    if not result.filepath.endswith("mp4"):
        media_resp = twitter_api.request(
            "media/upload", None, {"media": open(result.filepath, mode="rb").read()}
        )
    else:
        bytes_sent = 0
        total_bytes = os.path.getsize(result.filepath)
        file = open(result.filepath, "rb")
        init_req = twitter_api.request(
            "media/upload",
            {"command": "INIT", "media_type": "video/mp4", "total_bytes": total_bytes},
        )

        media_id = init_req.json()["media_id"]
        segment_id = 0

        while bytes_sent < total_bytes:
            chunk = file.read(4 * 1024 * 1024)
            twitter_api.request(
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
        media_resp = twitter_api.request(
            "media/upload", {"command": "FINALIZE", "media_id": media_id}
        )
    try:
        media = media_resp.json()
        media_id = media["media_id"]
        twitter_post = {
            "status": prompt.prompt,
            "media_ids": media_id,
        }
        twitter_api.request("statuses/update", twitter_post)
    except KeyError:
        try:
            logging.error(media_resp.text)
            admin(media_resp.text)
        except:  # pylint: disable=bare-except
            pass


if __name__ == "__main__":
    main()
