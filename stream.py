# utils.!/usr/bin/python3.9
import asyncio
import io
import logging
import time
import sys
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Optional
from PIL import Image
import uvloop
import better_imagegen as clipart
from utils import get_secret, timer

try:
    sys.path.append("./Real-ESRGAN")
    from realesrgan import RealESRGAN
except:
    RealESRGAN = None

fps = 60
dest = get_secret("YOUTUBE_URL")
silence = "-f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100"
pipe = f"""-y -thread_queue_size 1024 -analyzeduration 60 -f image2pipe -vcodec bmp -r 5 -i - -r {fps}"""
cmd = f"""ffmpeg -re {silence} \\
    {pipe} \\
    -f flv \\
    -pix_fmt yuvj420p \\
    -max_muxing_queue_size 1024
    -x264-params keyint=300:min-keyint=48:scenecut=-1 \\
    -b:a 128k \\
    -b:v 1000k \\
    -ar 44100 \\
    -acodec aac \\
    -vcodec libx264 \\
    -preset ultrafast \\
    -crf 28 \\
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
    exiting = False

    async def fps(self) -> None:
        while not self.exiting:
            now = time.time()
            self.frame_times = [t for t in self.frame_times if now - t <= 1]
            self.generator.frame_times = [
                t for t in self.generator.frame_times if now - t <= 5
            ]
            avg_fps = len(self.generator.frame_times) / 5
            logging.info(
                f"ffmpeg write fps: {len(self.frame_times)}; generate fps: {avg_fps}"
            )
            await asyncio.sleep(1)
            if round(time.time() - now, 4) > 1:
                logging.info("elapsed time between fps ticks: %.4f", time.time() - now)

    async def yolo(self) -> None:
        args = clipart.base_args.with_update(
            {"size": [640, 360]} if get_secret("UPSAMPLE") else {"size": [320, 180]}
        )
        self.generator = clipart.Generator(args)
        upsampler = RealESRGAN() if get_secret("UPSAMPLE") and RealESRGAN else None
        generate_task = asyncio.create_task(self.generator.generate(args))
        # whole thing to a_thread??
        generate_task.add_done_callback(log_task_result)
        await asyncio.sleep(1)
        ffmpeg_proc: Optional[asyncio.subprocess.Process] = None
        last_bytes: Optional[bytes] = None
        frame_times = []
        while not generate_task.done():
            # potentially, we could delay starting the stream until we have the first *two* frames,
            # and crossfade between them until the following frame is available
            # unfortunately it's not known in advance how long that will be?
            # but you could have a slightly choppy predictive algorithm, maybe smoothed by 0.5
            wait_start = time.time()
            try:
                logging.debug("waiting for next frame")
                image = await asyncio.wait_for(self.generator.image_queue.get(), 0.2)
                logging.info("got new bytes after %.4f", time.time() - wait_start)
                with timer("upsampling?"):
                    last_bytes = get_img_bytes(
                        upsampler.predict(image) if upsampler else image
                    )
                if not ffmpeg_proc:
                    logging.info("starting ffmpeg")
                    ffmpeg_proc = await asyncio.create_subprocess_exec(
                        *cmd.split(), stdin=PIPE
                    )
                    asyncio.create_task(self.fps())
            except asyncio.TimeoutError:
                logging.info(
                    "timed out waiting for new bytes after %.4f, reusing previous frame",
                    time.time() - wait_start,
                )
            if last_bytes and ffmpeg_proc:
                assert ffmpeg_proc.stdin
                ffmpeg_proc.stdin.write(last_bytes)
                await ffmpeg_proc.stdin.drain()
                frame_times.append(time.time())
                logging.info("wrote bytes to ffmpeg.")
        self.exiting = True
        logging.info("done")
        if ffmpeg_proc:
            try:
                await asyncio.wait_for(ffmpeg_proc.wait(), 2.0)
            except asyncio.TimeoutError:
                ffmpeg_proc.terminate()


if __name__ == "__main__":
    # run()
    # raise SystemExit
    try:
        uvloop.install()
        asyncio.run(Streamer().yolo())
    except BrokenPipeError:
        pass
