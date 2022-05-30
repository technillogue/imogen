#!/usr/bin/python3.9
import sys
import os
from subprocess import PIPE, Popen
from tqdm.notebook import tqdm
import numpy as np
from pathlib import Path
from PIL import Image

fps = 10
dest = os.environ["YOUTUBE_URL"] 
silence = "-f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100"
pipe = (
    f"""-y -thread_queue_size 1024 -f image2pipe -vcodec png -r {fps} -i - -r {fps}"""
)
cmd = f"""ffmpeg -re {silence} \\
    {pipe} \\
    -f flv \\
    -pix_fmt yuvj420p \\
    -x264-params keyint=48:min-keyint=48:scenecut=-1 \\
    -b:v 4500k \\
    -b:a 128k \\
    -ar 44100 \\
    -acodec aac \\
    -vcodec libx264 \\
    -preset veryslow \\
    -crf 28 \\
    {dest}
""".replace("\\\n", "").strip()

dir = "./output/2022-05-13T15:31:42.460104pink_elephant_in_spacepastel_fire_sculptured9f7/steps"


def run():
    p = Popen(
        cmd.split(),
        stdin=PIPE,
    )
    assert p.stdin

    for path in sorted(Path(dir).iterdir()):
        Image.open(str(path)).save(p.stdin, "PNG")
    p.stdin.close()
    p.wait()


def get_proc():
    return Popen(
        cmd.split(),
        stdin=PIPE,
    )


if __name__ == "__main__":
    try:
        run()
    except BrokenPipeError:
        pass
