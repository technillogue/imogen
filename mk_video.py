import os
from subprocess import PIPE, Popen
from tqdm.notebook import tqdm


def video(i: int, root: str = "") -> None:
    min_fps = 10
    max_fps = 30
    length = 15  # Desired video time in seconds

    frames = [
        Image.open(f"{root}/steps/{i:04}.png") for fname in sorted(os.listdir("steps"))
    ]
    # for a, b in zip(frames[:-1], frames[1:]):
    #   # interpolate?

    total_frames = len(frames)
    print("total frames: {total_frames}, fps: {fps}")
    # fps = last_frame/10
    fps = np.clip(total_frames / length, min_fps, max_fps)

    cmd = f"ffmpeg -y -f image2pipe -vcodec png -r {fps} -i - -vcodec libx264 -r {fps} -pix_fmt yuv420p -crf 17 -preset veryslow video.mp4"
    p = Popen(
        cmd.split(" "),
        stdin=PIPE,
    )
    for i, im in enumerate(tqdm(frames)):

        im.save(p.stdin, "PNG")
    p.stdin.close()

    print("The video is now being compressed, wait...")
    p.wait()
    print("The video is ready")
