# Copyright (c) 2022 Sylvie Liberman
import functools
import logging
import math
import os
import time
from contextlib import contextmanager
from typing import Iterator, Optional, cast
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F


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


@contextmanager
def timer(msg: str) -> Iterator:
    logging.debug("started %s", msg)
    start_time = time.time()
    yield
    logging.info("done %s after %.4f", msg, time.time() - start_time)


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


def resample(
    input: Tensor, size: tuple[int, int], align_corners: bool = True
) -> Tensor:
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


def resize_image(image: Image, out_size: tuple) -> Image:
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio) ** 0.5), round((area / ratio) ** 0.5)
    return image.resize(size, Image.LANCZOS)


# def add_xmp_data(filename):
#     imagen = ImgTag(filename=filename)
#     imagen.xmp.append_array_item(
#         libxmp.consts.XMP_NS_DC,
#         "creator",
#         "VQGAN+CLIP",
#         {"prop_array_is_ordered": True, "prop_value_is_array": True},
#     )
#     if args.prompts:
#         imagen.xmp.append_array_item(
#             libxmp.consts.XMP_NS_DC,
#             "title",
#             " | ".join(args.prompts),
#             {"prop_array_is_ordered": True, "prop_value_is_array": True},
#         )
#     else:
#         imagen.xmp.append_array_item(
#             libxmp.consts.XMP_NS_DC,
#             "title",
#             "None",
#             {"prop_array_is_ordered": True, "prop_value_is_array": True},
#         )
#     imagen.xmp.append_array_item(
#         libxmp.consts.XMP_NS_DC,
#         "i",
#         str(i),
#         {"prop_array_is_ordered": True, "prop_value_is_array": True},
#     )
#     imagen.xmp.append_array_item(
#         libxmp.consts.XMP_NS_DC,
#         "model",
#         nombre_modelo,
#         {"prop_array_is_ordered": True, "prop_value_is_array": True},
#     )
#     imagen.xmp.append_array_item(
#         libxmp.consts.XMP_NS_DC,
#         "seed",
#         str(seed),
#         {"prop_array_is_ordered": True, "prop_value_is_array": True},
#     )
#     imagen.xmp.append_array_item(
#         libxmp.consts.XMP_NS_DC,
#         "input_images",
#         str(input_images),
#         {"prop_array_is_ordered": True, "prop_value_is_array": True},
#     )
#     # for frases in args.prompts:
#     #    imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'Prompt' ,frases, {"prop_array_is_ordered":True, "prop_value_is_array":True})
#     imagen.close()


# def add_stegano_data(filename):
#     data = {
#         "title": " | ".join(args.prompts) if args.prompts else None,
#         "notebook": "VQGAN+CLIP",
#         "i": i,
#         "model": nombre_modelo,
#         "seed": str(seed),
#         "input_images": input_images,
#     }
#     lsb.hide(filename, json.dumps(data)).save(filename)
