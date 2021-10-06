import math
import torch
from torch import Tensor
from torch.nn import functional as F


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


def resample(input: Tensor, size: "list[int]", align_corners: bool = True) -> Tensor:
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


# def resize_image(image, out_size):
#     ratio = image.size[0] / image.size[1]
#     area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
#     size = round((area * ratio) ** 0.5), round((area / ratio) ** 0.5)
#     return image.resize(size, Image.LANCZOS)