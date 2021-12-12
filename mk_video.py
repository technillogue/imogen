#!/usr/bin/python3.9
import sys
import os
from subprocess import PIPE, Popen
from tqdm.notebook import tqdm
import numpy as np
from pathlib import Path
from PIL import Image
def video(root: str = "") -> None:
    min_fps = 10
    max_fps = 30
    length = 15  # Desired video time in seconds

    frames = (Path(root) / "steps").iterdir()
    # for a, b in zip(frames[:-1], frames[1:]):
    #   # interpolate?

    total_frames = len(list(frames))
    # fps = last_frame/10
    fps = np.clip(total_frames / length, min_fps, max_fps)

    print(f"total frames: {total_frames}, fps: {fps}")
    cmd = f"ffmpeg -y -f image2pipe -vcodec png -r {fps} -i - -vcodec libx264 -r {fps} -pix_fmt yuv420p -crf 17 -preset veryslow {root}/video.mp4"
    p = Popen(
        cmd.split(" "),
        stdin=PIPE,
    )
    assert p.stdin

    for path in sorted((Path(root) / "steps").iterdir()):
        Image.open(str(path)).save(p.stdin, "PNG")
    p.stdin.close()

    print("The video is now being compressed, wait...")
    p.wait()
    print("The video is ready")

# ffmpeg \                                         
#     -re \
#     -framerate 10 \
#     -stream_loop -1 \
#     -f image2 \
#     -i "progress.png" \
#     -c:v libx264 \
#     -preset superfast \
#     -tune zerolatency \
#     -pix_fmt yuv420p \
#     -s 1000x1000 \
#     -r 25 \
#     -f flv  "rtmp://live-fra.twitch.tv/app/STREAM_KEY"

if __name__=="__main__":
    try:
        video(sys.argv[1])
    except IndexError:
        video()
