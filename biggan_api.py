import math
from moviepy.editor import concatenate, ImageClip
import os
import platform
import subprocess
import random
import torch
# pip install pytorch-pretrained-biggan
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample, convert_to_images)
import numpy as np

model = BigGAN.from_pretrained('biggan-deep-512')
total_frames = 60


def convert_to_video(imgs):
    fps = 30
    clips = [ImageClip(m).set_duration(1/fps)
             for m in imgs]
    folder = "/".join(imgs[0].split("/")[:-1])
    video = concatenate(clips, method="compose")
    filename = '%s/video.mp4' % folder
    video.write_videofile(filename, fps=fps)
    return filename


def open_file(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def get_image(class_vector, noise_vector, truncation=0.5):
    class_vector = torch.from_numpy(class_vector)
    noise_vector = torch.from_numpy(noise_vector)

    # If you have a GPU, put everything on cuda
    # noise_vector = noise_vector.to('cuda')
    # class_vector = class_vector.to('cuda')
    # model.to('cuda')

    with torch.no_grad():
        output = model(noise_vector, class_vector, truncation)

    return convert_to_images(output)[0]


def get_classvector(num):
    class_vector = [[np.float32(0)] * 1000]
    class_vector[0][num] = np.float32(1.0)
    return np.array(class_vector)


def clamp(param, min_, max_):
    return max(min_, min(max_, param))


def generate_all_classes():
    for num in range(0, 1000):
        generate_class(num)


def generate_class(num):
    truncation = 0.5
    batch_size = 1

    for variation in range(1, 6):
        path = "images/class_%d" % num
        filename = "%s/%d.png" % (path, variation)
        if not os.path.exists(filename):
            noise_vector = truncated_noise_sample(truncation=truncation, batch_size=batch_size)
            class_vector = get_classvector(num)
            if not os.path.exists(path):
                os.makedirs(path)
            get_image(class_vector, noise_vector, truncation).save(filename)


def generate_random_morph():
    all = list(range(1, 1001))
    num1 = random.choice(all)
    all.remove(num1)
    num2 = random.choice(all)
    noise_vector = truncated_noise_sample(truncation=0.5, batch_size=1)
    path = "animations/class_%d_%d" % (num1, num2)

    if not os.path.exists(path):
        os.makedirs(path)

    make_frames(list(range(total_frames + 1)), total_frames, num1, num2, noise_vector, path, )

    open_file(path)


def generate_random_morph_sequence(count, silent=False):
    noise_vector = truncated_noise_sample(truncation=0.5, batch_size=1)

    all = list(range(1, 1001))
    nums = []
    for i in range(count):
        num = random.choice(all)
        all.remove(num)
        nums.append(num)
    nums.append(nums[0])

    path = "animations/class_%s" % "_".join([str(a) for a in nums])

    if not os.path.exists(path):
        os.makedirs(path)

    frames = []
    for i in range(len(nums) - 1):
        num1 = nums[i]
        num2 = nums[i + 1]
        frames += make_frames(list(range(total_frames + 1)), total_frames, num1, num2, noise_vector, path,
                              i * total_frames)

    if not silent:
        open_file(path)

    convert_to_video(frames)


def make_frames(c, total_frames, num1, num2, noise_vector, path, start=0):
    frames = []
    for i in c:
        frame = make_frame(i, total_frames, num1, num2, noise_vector, path, start)
        if frame:
            frames.append(frame)
    return frames


def ease(t):
    return t


def make_frame(i, total_frames, num1, num2, noise_vector, path, start=0):
    perc = i / total_frames
    perc1 = ease(1 - perc)
    perc2 = ease(perc)
    class_vector = [[np.float32(0)] * 1000]
    class_vector[0][num1] = np.float32(perc1)
    class_vector[0][num2] = np.float32(perc2)
    class_vector = np.array(class_vector)
    filename = "%s/%d.png" % (path, start + i)
    if not os.path.exists(filename):  # avoids making frame that already exists
        get_image(class_vector, noise_vector).save(filename)
        print(filename)
        return filename
    return None


if __name__ == '__main__':
    # generate_all_classes() #  this will generate one image of each class
    generate_random_morph_sequence(10, True)
