#!/usr/bin/python3.8
import argparse
import io

import imageio
import kornia.augmentation as K
import numpy as np
import torch

# from IPython import display
from omegaconf import OmegaConf
from PIL import Image, ImageFile

sys.path.append("./taming-transformers")
from taming.models import cond_transformer, vqgan
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

# other image stuff?
from CLIP import clip
from image_utils import resample

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        (input,) = ctx.saved_tensors
        return (
            grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0),
            None,
            None,
        )


clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x, codebook):
    d = (
        x.pow(2).sum(dim=-1, keepdim=True)
        + codebook.pow(2).sum(dim=1)
        - 2 * x @ codebook.T
    )
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode="border"),
            K.RandomPerspective(0.2, p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
        )
        self.noise_factor = 0.1

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(
                torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size
            )
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_factor:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_factor)
            batch = batch + facs * torch.randn_like(batch)
        return batch


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == "taming.models.vqgan.VQModel":
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == "taming.models.cond_transformer.Net2NetTransformer":
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f"unknown model type: {config.model.target}")
    del model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio) ** 0.5), round((area / ratio) ** 0.5)
    return image.resize(size, Image.LANCZOS)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1.0, stop=float("-inf"), dwelt=0, tag=""):
        super().__init__()
        self.dwelt = dwelt
        self.register_buffer("embed", embed)
        self.register_buffer("weight", torch.as_tensor(weight))
        self.register_buffer("stop", torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return (
            self.weight.abs()
            * replace_grad(dists, torch.maximum(dists, self.stop)).mean()
        )


def generate(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
    perceptor = (
        clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
    )
    cut_size = perceptor.visual.input_resolution
    e_dim = model.quantize.e_dim
    f = 2 ** (model.decoder.num_resolutions - 1)
    make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
    n_toks = model.quantize.n_e
    toksX, toksY = args.size[0] // f, args.size[1] // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]


if args.init_image:
    pil_image = Image.open(args.init_image).convert("RGB")
    pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
    z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
else:
    one_hot = F.one_hot(
        torch.randint(n_toks, [toksY * toksX], device=device), n_toks
    ).float()
    if is_gumbel:
        z = one_hot @ model.quantize.embed.weight
    else:
        one_hot = F.one_hot(
            torch.randint(n_toks, [toksY * toksX], device=device), n_toks
        ).float()
        z = one_hot @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    z_orig = z.clone()
    z.requires_grad_(True)
    opt = optim.Adam([z], lr=args.step_size)

    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    prompt_modules = []

    print(f"using text prompt {args.prompts} and image prompt {args.image_prompts}")

    for prompt in args.prompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        prompt_modules.append(PromptModule(embed, weight, stop).to(device))

    for prompt in args.image_prompts:
        path, weight, stop = parse_prompt(prompt)
        img = resize_image(Image.open(fetch(path)).convert("RGB"), (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()

        prompt_modules.append(PromptModule(embed, weight, stop).to(device))

    for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        prompt_modules.append(PromptModule(embed, weight).to(device))

    def synth(z):
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(
            3, 1
        )
        return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)


@torch.no_grad()
def checkin(i, losses):
    losses_str = ", ".join(f"{loss.item():g}" for loss in losses)
    tqdm.write(f"i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}")
    # out = synth(z)
    # TF.to_pil_image(out[0].cpu()).save('progress.png')
    # img = display.Image('progress.png')
    # handle.update(img)
    # display.display(img)


prompt_queue = list(texts)
first_text = prompt_queue.pop(0)
second_text = prompt_queue.pop(0)

prompts = [
    Prompt(embed(first_text), 1.0).to(device),
    Prompt(embed(second_text), weight=0.0).to(device),
]


# csv?
# text digest, weight, text


def describe_prompt(text: str, prompt: Prompt) -> None:
    print(f"{text}: dwell {prompt.dwelt}, weight {prompt.weight}. ", end="")

    @torch.no_grad()
    def crossfade_prompts(prompts, fade=300, dwell=300) -> list:
        global first_text, second_text
        # queue = open("queue").readlines()
        if prompts[0].dwelt < dwell:
            prompts[0].dwelt += 1
            print("dwell: ", prompts[0].dwelt)
        elif prompts[0].weight > 0 and len(prompts) >= 2:
            first, second = prompts
            waning_weight = float(first.weight) - 1 / fade
            waxing_weight = min(1.0, float(second.weight) + 1 / fade)
            prompts[0] = Prompt(first.embed, waning_weight, dwelt=first.dwelt)
            prompts[1] = Prompt(second.embed, waxing_weight, dwelt=second.dwelt)
        else:
            prompts.pop(0)
            next_text = prompt_queue.pop(0)
            print("next text: ", next_text)
            first_text, second_text = second_text, next_text
            prompts.append(Prompt(embed(next_text), weight=0).to(device))
        if i % args.display_freq == 0:
            describe_prompt(first_text, prompts[0])
            describe_prompt(second_text, prompts[1])
            print()
        return prompts

    def ascend_txt():
        global i
        out = synth(z)
        iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

        result = []

        if args.init_weight:
            result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)

        for prompt in crossfade_prompts(prompts, args.fade, args.dwell):
            result.append(prompt(iii))
        with torch.no_grad():
            img = np.array(
                out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8)
            )[:, :, :]
            img = np.transpose(img, (1, 2, 0))
            filename = f"{root}/steps/{i:04}.png"
            imageio.imwrite(filename, np.array(img))
            # img = display.Image(f"{root}/steps/{i:04}.png")
            # handle.update(img)

        return result

    def train(i):
        opt.zero_grad()
        lossAll = ascend_txt(i)
        if i % args.display_freq == 0:
            checkin(i, lossAll)
        loss = sum(lossAll)
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.copy_(z.maximum(z_min).minimum(z_max))

    i = 0
    start = time.time()
    try:
        with tqdm() as pbar:
            try:
                train(i)
            except IndexError:
                break
            if i == max_iterations:
                break
            i += 1
            pbar.update()
    except KeyboardInterrupt:
        pass


base_args = BetterNamespace(
    prompts=[
        "the ocean is on fire",
        "an abstract painting by a talented artist",
    ],
    image_prompts=[],  # "dracula.jpeg"],
    noise_prompt_seeds=[],
    noise_prompt_weights=[],
    size=[780, 480],
    init_image=None,
    init_weight=0.0,
    clip_model="ViT-B/32",
    vqgan_config="vqgan_imagenet_f16_16384.yaml",
    vqgan_checkpoint="vqgan_imagenet_f16_16384.ckpt",
    step_size=0.1,
    cutn=64,
    cut_pow=1.0,
    display_freq=10,
    seed=0,
    max_iterations=200,
    name="container",
    fade=100,  # @param {type:"number"}
    dwell=100,  # @param {type: "number"}
)
