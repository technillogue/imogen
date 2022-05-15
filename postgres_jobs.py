#!/usr/bin/python3.9
# Copyright (c) 2022 Sylvie Liberman
# pylint: disable=subprocess-run-check
import cProfile
import dataclasses
import json
import logging
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import psycopg
import redis
import requests
import TwitterAPI as t
from psycopg.rows import class_row

import better_imagegen as clipart

# import feedforward
import mk_video
import utils

hostname = socket.gethostname()
logging.getLogger().setLevel("DEBUG")
twitter_api = t.TwitterAPI(
    *utils.get_secret("TWITTER_CREDS").split(","),
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

password, rest = utils.get_secret("REDIS_URL").removeprefix("redis://:").split("@")
host, port = rest.split(":")
r = redis.Redis(host=host, port=int(port), password=password)


def admin(msg: str) -> None:
    """send a message to admin"""
    logging.info(msg)
    requests.post(
        f"{admin_signal_url}/admin",
        params={"message": str(msg)},
    )


def stop() -> None:
    "check envvars if we should exit depending on where we're running, or sleep"
    paid = "" if os.getenv("FREE") else "paid "
    logging.debug("stopping")
    if os.getenv("POWEROFF"):
        admin(
            f"\N{cross mark}{paid}\N{frame with picture}\N{construction worker}\N{high voltage sign}\N{downwards black arrow} {hostname}"
        )
        subprocess.run(["sudo", "poweroff"])
    elif os.getenv("EXIT"):
        admin(
            f"\N{cross mark}{paid}\N{frame with picture}\N{construction worker}\N{sleeping symbol} {hostname}"
        )
        sys.exit(0)
    else:
        time.sleep(15)


# this isn't used?
def maybe_scale_in(conn: psycopg.Connection) -> None:
    "check the ratio of paid prompts to paid workers and potentially stop"
    if not os.getenv("EXIT_ON_LOAD"):
        return
    workers = conn.execute(
        "select count(distinct hostname) + 1 from prompt_queue where status='assigned' and selector=%s and paid=true",
        [os.getenv("SELECTOR", "")],
    ).fetchone()[0]
    queue_empty = conn.execute(
        "SELECT count(id)=0 FROM prompt_queue WHERE status='pending'"
    ).fetchone()[0]
    paid_queue_size = conn.execute(
        "SELECT count(id) AS len FROM prompt_queue WHERE status='pending' OR status='assigned' AND paid=TRUE;"
    ).fetchone()[0]
    if queue_empty:
        admin(f"\N{scales}\N{chart with downwards trend}\N{octagonal sign} {hostname}")
        sys.exit(0)
    if workers == 1:
        # nobody else has taken assignments, we just finished ours
        return
    if paid_queue_size / workers < 5 or workers > 6:
        # target metric: latency under 10 min for paid images
        # images take ~2min
        # if there's less than five items per worker, we aren't needed
        # even if there's 25 items, we still don't want more than five workers
        admin(
            f"paid queue size: {paid_queue_size}. workers: {workers}. load: {paid_queue_size / workers}. exiting {hostname}"
        )
        sys.exit(0)


@dataclasses.dataclass
class Prompt:
    "holds database result with prompt information"
    prompt_id: int
    prompt: str
    url: str
    slug: str = ""
    params: str = ""
    inserted_ts: str = ""
    param_dict: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            self.param_dict = json.loads(self.params or "{}")
            assert isinstance(self.param_dict, dict)
        except (json.JSONDecodeError, AssertionError):
            self.param_dict = {}
        self.slug = clipart.mk_slug(self.prompt, self.inserted_ts)


@dataclasses.dataclass
class Result:
    "info after generated a prompt used to update database"
    elapsed: int
    loss: float
    seed: str
    filepath: str


def get_prompt(conn: psycopg.Connection) -> Optional[Prompt]:
    "try to get a prompt and mark it as assigned if possible"
    # mark prompts that have been assigned for more than 10 minutes as unassigned
    conn.execute(
        """UPDATE prompt_queue SET status='pending', assigned_at=null
        WHERE status='assigned' AND assigned_at  < (now() - interval '10 minutes');"""
    )  # maybe this is a trigger
    # try to select something
    maybe_id = conn.execute(
        f"""SELECT id FROM prompt_queue WHERE status='pending'
        AND selector=%s AND paid=%s ORDER BY signal_ts ASC LIMIT 1;""",
        [os.getenv("SELECTOR", ""), not bool(os.getenv("FREE"))],
    ).fetchone()
    if not maybe_id:
        return None
    prompt_id = maybe_id[0]
    cursor = conn.cursor(row_factory=class_row(Prompt))
    logging.info("getting")
    # mark it as assigned, returning only if it got updated
    maybe_prompt = cursor.execute(
        "UPDATE prompt_queue SET status='assigned', assigned_at=now(), hostname=%s WHERE id = %s RETURNING id AS prompt_id, prompt, params, url, inserted_ts::text;",
        [hostname, prompt_id],
    ).fetchone()
    logging.info("set assigned")
    return maybe_prompt


def main() -> None:
    "setup, get prompts, handle them, mark as uploading, upload, mark done"
    Path("./input").mkdir(exist_ok=True)
    admin(f"\N{artist palette}\N{construction worker}\N{hiking boot} {hostname}")
    logging.info("starting postgres_jobs on %s", hostname)
    # clear failed instances
    # try to get an id. if we can't, there's no work, and we should stop
    # try to claim it. if we can't, someone else took it, and we should try again
    # generate the prompt
    backoff = 60.0
    generator = None
    conn = psycopg.connect(utils.get_secret("DATABASE_URL"), autocommit=True)
    # catch some database connection errors
    try:
        while 1:
            # try to claim
            prompt = get_prompt(conn)
            if not prompt:
                time.sleep(60)
                prompt = get_prompt(conn)
                if not prompt:
                    stop()
                    continue
            logging.info("got prompt: %s", prompt)
            try:
                generator, result = handle_item(generator, prompt)
                # success
                start_post = time.time()
                fmt = """UPDATE prompt_queue SET status='uploading', loss=%s, elapsed_gpu=%s, filepath=%s, seed=%s WHERE id=%s;"""
                params = [
                    result.loss,
                    result.elapsed,
                    prompt.slug + "." + result.filepath.split(".")[-1],
                    result.seed,
                    prompt.prompt_id,
                ]
                logging.info("set uploading %s", prompt)
                conn.execute(fmt, params)
                post(result, prompt)
                conn.execute(
                    "UPDATE prompt_queue SET status='done' WHERE id=%s",
                    [prompt.prompt_id],
                )
                logging.info("set done, poasting time: %s", time.time() - start_post)
                backoff = 60
            except Exception as e:  # pylint: disable=broad-except
                logging.info("caught exception")
                error_message = traceback.format_exc()
                if prompt:
                    admin(repr(prompt))
                logging.error(error_message)
                admin(error_message)
                if "out of memory" in str(e):
                    sys.exit(137)
                conn.execute(
                    "UPDATE prompt_queue SET status='pending', errors=errors+1 WHERE id=%s",
                    [prompt.prompt_id],
                )
                time.sleep(backoff)
                backoff *= 1.5
            maybe_scale_in(conn)
    finally:
        conn.close()


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
Gen = Optional[clipart.Generator]


def handle_item(generator: Gen, prompt: Prompt) -> tuple[Gen, Result]:
    "finagle settings, generate it depending on settings, make a video if appropriate"
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
            update = {"prompts": maybe_prompt_list, "max_iterations": 1000}
        else:
            update = {"text": prompt.prompt, "max_iterations": 221}
        args = clipart.base_args.with_update(update)
    if init_image := prompt.param_dict.get("init_image"):
        # download the image from redis
        open(init_image, "wb").write(r[init_image])
    for image_prompt in prompt.param_dict.get("image_prompts", []):
        open(image_prompt, "wb").write(r[image_prompt])

    prompt.param_dict["video"] = video
    args = args.with_update(prompt.param_dict | {"slug": prompt.slug})
    logging.info(args)
    path = f"output/{prompt.slug}"
    # feedforward_path = ""
    start_time = time.time()
    # if prompt.param_dict.get("feedforward"):
    #     feedforward_path = f"results/single/{prompt.slug}/progress.png"
    #     loss = feedforward.generate(prompt.prompt)
    # elif prompt.param_dict.get("feedforward_fast"):
    #     feedforward_path = f"results/single/{prompt.slug}.png"
    #     feedforward.generate_forward(prompt.prompt, out_path=feedforward_path)
    #     loss = -1
    if 1:  # else: # pylint: disable=using-constant-test
        if not generator or not generator.same_model(args):
            # hopefully purge memory used by previous model
            del generator
            generator = clipart.Generator(args)
        if args.profile:
            with cProfile.Profile() as profiler:
                loss, seed = generator.generate(args)
            profiler.dump_stats(f"profiling/{clipart.version}.stats")
            logging.info("generated with stats")
        else:
            loss, seed = generator.generate(args)
            logging.info("generated")
            if video:
                mk_video.video(path)
    fname = "video.mp4" if video else "progress.png"
    # return the generator so it can be reused
    return generator, Result(
        elapsed=round(time.time() - start_time),
        # filepath=feedforward_path or f"{path}/{fname}",
        filepath=f"{path}/{fname}",
        loss=round(loss, 4),
        seed=str(seed),
    )


def post(result: Result, prompt: Prompt) -> None:
    "upload to signal bot imogen, then upload to s3"
    minutes, seconds = divmod(result.elapsed, 60)
    f = open(result.filepath, mode="rb")
    message = f"{prompt.prompt}\nTook {minutes}m{seconds}s to generate,"
    if result.loss:
        message += f"{result.loss} loss,"
    message += f" v{clipart.version}."
    for i in range(3):
        try:
            resp = requests.post(
                f"{prompt.url or admin_signal_url}/attachment",
                params={"message": message, "id": str(prompt.prompt_id)},
                files={"image": f},
            )
            logging.info(resp)
            break
        except requests.RequestException:
            logging.info("pausing before retry")
            time.sleep(i)
    if not prompt.param_dict.get("nopost") and utils.get_secret("TWITTER"):
        post_tweet(result, prompt)
    bearer = "Bearer " + utils.get_secret("SUPABASE_API_KEY")
    mime = "video/mp4" if result.filepath.endswith("mp4") else "image/png"
    requests.post(
        f"https://mcltajcadcrkywecsigc.supabase.in/storage/v1/object/imoges/{prompt.slug}.png",
        headers={"Authorization": bearer, "Content-Type": mime},
        data=open(result.filepath, mode="rb").read(),
    )
    os.remove(result.filepath)
    # can be retrieved with
    # slug = prompt_queue.filepath.split("/")[1] # bc slug= the directory in filepath
    # requests.get(
    # f"https://mcltajcadcrkywecsigc.supabase.in/storage/v1/object/public/imoges/{prompt.slug}.png"
    # )


def retry_uploads(limit: int = 10, recent: bool = False) -> None:
    """retry uploading prompts that are available locally but never got uploaded.
    this is only really run manually, and doesn't make as much sense with ephemeral pods"""
    conn = psycopg.connect(utils.get_secret("DATABASE_URL"), autocommit=True)
    q = conn.execute(
        "select id, url, filepath from prompt_queue where status='uploading' and hostname=%s "
        f"order by id {'desc' if recent else 'asc'} limit %s",
        [hostname, limit],
    )
    try:
        for prompt_id, url, filepath in q:
            try:
                f = open(filepath, mode="rb")
            except FileNotFoundError:
                continue
            try:
                _url = f"{url or admin_signal_url}/attachment"
                resp = requests.post(
                    _url, params={"id": str(prompt_id)}, files={"image": f}
                )
                logging.info(resp)
                if resp.status_code == 200:
                    conn.execute(
                        "update prompt_queue set status='done' where id=%s", [prompt_id]
                    )
            except:  # pylint: disable=bare-except
                continue
    finally:
        conn.close()


def post_tweet(result: Result, prompt: Prompt) -> None:
    "post tweet, either all at once for images or in chunks for videos"
    logging.info("uploading to twitter")
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
            logging.error("couldn't send to admin")


if __name__ == "__main__":
    main()
