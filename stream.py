#!/usr/bin/python3.9
import asyncio
import io
import os
import time
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Optional
from PIL import Image
import better_imagegen as clipart
import logging

fps = 30
dest = os.getenv("YOUTUBE_URL") or open("youtube_url").read()
silence = "-f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100"
pipe = f"""-y -thread_queue_size 1024 -analyzeduration 60 -f image2pipe -vcodec bmp -r 5 -i - -r {fps}"""
cmd = f"""ffmpeg -re {silence} \\
    {pipe} \\
    -f flv \\
    -pix_fmt yuvj420p \\
    -max_muxing_queue_size 512
    -x264-params keyint=48:min-keyint=48:scenecut=-1 \\
    -b:v 1000k \\
    -b:a 128k \\
    -ar 44100 \\
    -acodec aac \\
    -vcodec libx264 \\
    -preset ultrafast \\
    -crf 50 \\
    {dest}
""".replace(
    "\\\n", ""
).strip()


def run() -> None:
    d = "output/2022-05-13T16:11:18.947478pink_elephant_in_spacepastel_fire_sculpture8688"
    imgs = [
        get_img_bytes(Image.open(str(path)))
        for path in sorted(Path(d).glob("**/*png"))[50:150]
    ]
    p = Popen(
        cmd.split(),
        stdin=PIPE,
    )
    assert p.stdin
    # for path in sorted(Path("output").glob("**/*.png")):
    #     Image.open(str(path)).save(p.stdin, format="PNG")

    for epoch in range(100):
        for img in imgs:
            p.stdin.write(img)
            p.stdin.flush()
    p.stdin.close()
    p.wait()


def get_proc() -> Popen:
    return Popen(
        cmd.split(),
        stdin=PIPE,
    )


def get_img_bytes(img: Image) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="bmp")
    buffer.seek(0)
    return buffer.read()


def log_task_result(
    task: asyncio.Task,
) -> None:
    """
    Done callback which logs task done result
    args:
        task (asyncio.task): Finished task
    """
    name = task.get_name() + "-" + getattr(task.get_coro(), "__name__", "")
    try:
        result = task.result()
        logging.info("final result of %s was %s", name, result)
    except asyncio.CancelledError:
        logging.info("task %s was cancelled", name)
    except Exception:  # pylint: disable=broad-except
        logging.exception("%s errored", name)


class Streamer:
    frame_times: list[float] = []

    async def fps(self):
        while 1:
            now = time.time()
            self.frame_times = [t for t in self.frame_times if now - t <= 1]
            logging.info(f"fps: {len(self.frame_times)}")
            await asyncio.sleep(1)
            if round(time.time() - now, 4) > 1:
                logging.info("elapsed time between fps ticks: %.4f", time.time() - now)

    async def yolo(self) -> None:
        args = clipart.base_args
        generator = clipart.Generator(args)
        generate_task = asyncio.create_task(generator.generate(args))
        generate_task.add_done_callback(log_task_result)
        # to_thread??
        print("sleeping")
        await asyncio.sleep(1)
        ffmpeg_proc: Optional[asyncio.subprocess.Process] = None
        last_bytes: Optional[bytes] = None
        frame_times = []
        while not generate_task.done():
            # potentially, we could delay starting the stream until we have the first *two* frames,
            # and crossfade between them until the following frame is available
            # unfortunately it's not known in advance how long that will be?
            # but you could have a slightly choppy predictive algorithm, maybe smoothed by 0.5
            try:
                logging.info("waiting for next frame")
                wait_start = time.time()
                last_bytes = get_img_bytes(
                    await asyncio.wait_for(generator.image_queue.get(), 0.2)
                )
                logging.info("got new bytes after %.4f", time.time() - wait_start)
                if not ffmpeg_proc:
                    logging.info("starting ffmpeg")
                    ffmpeg_proc = await asyncio.create_subprocess_exec(
                        *cmd.split(), stdin=PIPE
                    )
                    asyncio.create_task(self.fps())
            except asyncio.TimeoutError:
                logging.info("timed out waiting for new bytes, reusing previous frame")
            if last_bytes and ffmpeg_proc:
                assert ffmpeg_proc.stdin
                ffmpeg_proc.stdin.write(last_bytes)
                await ffmpeg_proc.stdin.drain()
                frame_times.append(time.time())
                logging.info("wrote bytes to ffmpeg.")
        logging.info("done")
        if ffmpeg_proc:

            await ffmpeg_proc.wait()


if __name__ == "__main__":
    # run()
    # raise SystemExit
    try:
        asyncio.run(Streamer().yolo())
    except BrokenPipeError:
        pass
