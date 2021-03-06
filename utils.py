# Copyright (c) 2022 Sylvie Liberman
import math
import os
import functools
import logging
from typing import Optional, cast
import torch
from torch import Tensor
from torch.nn import functional as F
from PIL import Image


@functools.cache  # don't load the same env more than once?
def load_secrets(env: Optional[str] = None, overwrite: bool = False) -> None:
    if not env:
        env = os.environ.get("ENV", "dev")
    try:
        logging.info("loading secrets from %s_secrets", env)
        secrets = [
            line.strip().split("=", 1)
            for line in open(f"{env}_secrets")
            if line and not line.startswith("#")
        ]
        can_be_a_dict = cast(list[tuple[str, str]], secrets)
        if overwrite:
            new_env = dict(can_be_a_dict)
        else:
            new_env = (
                dict(can_be_a_dict) | os.environ
            )  # mask loaded secrets with existing env
        os.environ.update(new_env)
    except FileNotFoundError:
        pass


# TODO: split this into get_flag and get_secret; move all of the flags into fly.toml;
# maybe keep all the tomls and dockerfiles in a separate dir with a deploy script passing --config and --dockerfile explicitly
def get_secret(key: str, env: Optional[str] = None) -> str:
    try:
        secret = os.environ[key]
    except KeyError:
        load_secrets(env)
        secret = os.environ.get(key) or ""  # fixme
    if secret.lower() in ("0", "false", "no"):
        return ""
    return secret


def sinc(x: Tensor) -> Tensor:
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x: Tensor, a: int) -> Tensor:
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio: float, width: int) -> Tensor:
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0.0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input: Tensor, size: tuple[int, int], align_corners: bool = True) -> Tensor:
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        ramp_res = ramp(dh / h, 2)
        kernel_h = lanczos(ramp_res, 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), "reflect")
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), "reflect")
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode="bicubic", align_corners=align_corners)


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio) ** 0.5), round((area / ratio) ** 0.5)
    return image.resize(size, Image.LANCZOS)
