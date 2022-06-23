#!/usr/bin/python3.9
# Copyright (c) 2022 Sylvie Liberman
# Copyright (c) 2021 Katherine Crowson
import argparse
import asyncio
import logging
import pdb
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Union
import aioredis
import kornia.augmentation as K
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageFile
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm
import utils
from CLIP import clip
from utils import resample, resize_image

sys.path.append("./taming-transformers")
from taming.models import cond_transformer, vqgan

sys.path.append("./predict")
from likely_trainer import Likely, Gaussify

ImageFile.LOAD_TRUNCATED_IMAGES = True

version = "0.1"
logger = logging.getLogger()
logger.setLevel("DEBUG")
fmt = logging.Formatter("{levelname} {module}:{lineno}: {message}", style="{")
console_handler = logging.StreamHandler()
console_handler.setLevel("INFO")
console_handler.setFormatter(fmt)
logger.addHandler(console_handler)


def mk_slug(text: Union[str, list[str]], time: str = "") -> str:
    "strip offending charecters"
    really_time = time if time else datetime.now().isoformat()
    text = really_time + "".join(text).encode("ascii", errors="ignore").decode()
    return (
        "".join(c if (c.isalnum() or c in "._") else "_" for c in text)[:200]
        + hex(hash(text))[-4:]
    )


class ReplaceGrad(torch.autograd.Function):
    # replaces gradients used for the automatic gradient descent
    @staticmethod
    def forward(ctx, x_forward, x_backward):  # type: ignore
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):  # type: ignore
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    # i don't fully understand what this does
    @staticmethod
    def forward(ctx, input, min, max):  # type: ignore
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):  # type: ignore
        (input,) = ctx.saved_tensors
        return (
            grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0),
            None,
            None,
        )


clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x: Tensor, codebook: Tensor) -> Tensor:
    # https://arxiv.org/abs/1711.00937 is one of the papers introducing vector quantization
    d = (
        x.pow(2).sum(dim=-1, keepdim=True)
        + codebook.pow(2).sum(dim=1)
        - 2 * x @ codebook.T
    )
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class MakeCutouts(nn.Module):
    def __init__(self, cut_size: Any, cutn: int, cut_pow: float = 1.0) -> None:
        super().__init__()
        # size the cuts are resampled to (CLIP's input resolution)
        self.cut_size = cut_size
        # number of cuouts
        self.cutn = cutn
        # exponent for cut size to scale how big/small they are
        self.cut_pow = cut_pow
        # add some noise to the cutouts for cronchier gradients
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode="border"),
            K.RandomPerspective(0.2, p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
        )
        self.noise_factor = 0.1

    def forward(self, input: Tensor) -> Tensor:
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            # pick a size smaller than the max size and bigger than the min size
            size = int(
                torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size
            )
            # choose offset
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            # resample to CLIP's input resolution
            resampled = resample(cutout, (self.cut_size, self.cut_size))
            cutouts.append(resampled)
        # run each cutout through those augmentations
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_factor:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_factor)
            batch = batch + facs * torch.randn_like(batch)
        return batch


def load_vqgan_model(config_path: str, checkpoint_path: str) -> "VQModel":
    "load the VQGAN model from a checkpoint path and config path"
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


class Prompt(nn.Module):
    def __init__(
        self,
        embed: Tensor,  # Tensor[1, 512]
        weight: float = 1.0,
        stop: float = float("-inf"),
        dwelt: int = 0,
        tag: str = "",
    ) -> None:
        super().__init__()
        self.tag = tag
        self.dwelt = dwelt
        self.register_buffer("embed", embed)
        self.register_buffer("weight", torch.as_tensor(weight))
        self.register_buffer("stop", torch.as_tensor(stop))

    def forward(self, image_embedding: Tensor) -> Tensor:
        # image_embedding: Tensor[64, 512]
        # euclidian norm
        image_normed = F.normalize(image_embedding.unsqueeze(1), dim=2)
        text_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        # take cosine distance
        dists = image_normed.sub(text_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return (
            self.weight.abs()
            * replace_grad(dists, torch.maximum(dists, self.stop)).mean()
        )

    def describe(self) -> None:
        print(repr(self), end="")

    def __repr__(self) -> str:
        return f"{self.tag}: dwell {self.dwelt}, weight {self.weight}"


class ReactionPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.predictor = torch.load(
            "predict/likely.pth",
            map_location="cuda:0" if torch.cuda.is_available() else "cpu",
        )

    def forward(self, generated_image_embedding: Tensor) -> Tensor:
        # image_embedding is usually [64, 512]
        # predictor returns 1 if it would get reactions, but we want 0 to be good
        return 1 - self.predictor.predict_wide(generated_image_embedding).mean()


if utils.get_secret("REDIS_URL"):
    redis = aioredis.from_url(utils.get_secret("REDIS_URL"))
else:
    logging.warning("no redis")
    redis = None


class Generator:
    "class for holding onto the models"
    likely_loss: Optional[ReactionPredictor] = None

    def __init__(self, args: "BetterNamespace") -> None:
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info("Using device: %s", self.device)
        self.model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(
            self.device
        )
        logging.info("loaded model")
        # perceptor is CLIP, it calculates the loss between generated image and text prompt
        self.perceptor = (
            clip.load(args.clip_model, jit=False)[0]
            .eval()
            .requires_grad_(False)
            .to(self.device)
        )
        self.image_queue: asyncio.Queue[Image] = asyncio.Queue()
        self.frame_times: list[float] = []

    def same_model(self, new_args: "BetterNamespace") -> bool:
        "are the new args the same model as we have?"
        return (
            self.args.vqgan_checkpoint == new_args.vqgan_checkpoint
            and self.args.vqgan_config == new_args.vqgan_config
            and self.args.clip_model == new_args.clip_model
        )

    def embed(self, text: str) -> Tensor:  # Tensor[1, 512]
        "CLIP embed the text, stripping url"

        def no_url() -> Iterable[str]:
            "strip urls"
            for word in text.split(" "):
                scheme = word.startswith("http")
                twt = "twitter.com" in word
                if twt or scheme and "://" in word:
                    continue
                yield word

        cleantext = " ".join(no_url())
        return self.perceptor.encode_text(
            clip.tokenize(cleantext, truncate=True).to(self.device)
        ).float()

    def synth(self, z: Tensor) -> Tensor:
        "turn z into an image by vector quantizing then decoding"
        start = time.time()
        logging.debug("in synth")
        z_q = vector_quantize(
            z.movedim(1, 3), self.model.quantize.embedding.weight
        ).movedim(3, 1)
        result = clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)
        logging.debug("synth, elapsed: ", time.time() - start)
        return result

    async def generate(self, args: "BetterNamespace") -> tuple[float, int]:
        "actually generate an image using args"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cut_size = self.perceptor.visual.input_resolution
        embedding_dimension = self.model.quantize.e_dim
        f = 2 ** (self.model.decoder.num_resolutions - 1)
        make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
        n_toks = self.model.quantize.n_e
        toksX, toksY = args.size[0] // f, args.size[1] // f
        sideX, sideY = toksX * f, toksY * f
        z_min = self.model.quantize.embedding.weight.min(dim=0).values[
            None, :, None, None
        ]
        z_max = self.model.quantize.embedding.weight.max(dim=0).values[
            None, :, None, None
        ]
        if args.likely and not self.likely_loss:
            self.likely_loss = ReactionPredictor()

        if args.seed is not None:
            seed = args.seed
            torch.manual_seed(int(seed))
        else:
            seed = torch.seed()

        # z, the tensor representing the image in progress, is initialized as either the init image we got, or noise
        if args.init_image:
            pil_image = Image.open(args.init_image).convert("RGB")
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            z, *_ = self.model.encode(
                TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1
            )
        else:
            # start with noise
            one_hot = F.one_hot(
                torch.randint(n_toks, [toksY * toksX], device=device), n_toks
            ).float()
            z = one_hot @ self.model.quantize.embedding.weight
            z = z.view([-1, toksY, toksX, embedding_dimension]).permute(0, 3, 1, 2)
        z_orig = z.clone()
        z.requires_grad_(True)
        # lr is the learning rate for our optimizer
        # https://arxiv.org/abs/1412.6980
        opt = optim.Adam([z], lr=args.step_size)

        logging.info(
            f"using text prompt {args.prompts} and image prompt {args.image_prompts}"
        )

        slug = args.slug or mk_slug(args.prompts)
        try:
            (Path("output") / slug / "steps").mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass

        # this uses global z
        @torch.no_grad()
        def checkin(i: int, losses: "list[Tensor]") -> None:
            "log loss and save an image"
            losses_str = ", ".join(f"{torch.mean(loss).item():g}" for loss in losses)
            tqdm.write(f"i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}")
            out = self.synth(z)
            TF.to_pil_image(out[0].cpu()).save(f"output/{slug}/progress.png")

        # torch.Tensor([[1.0] * 512])
        prompts = [
            Prompt(self.embed(text), weight=weight, tag=text).to(device)
            for text, weight in zip(args.prompts, (1.0, 0.0))
        ]
        prompt_queue = args.prompts[2:]
        is_crossfade = len(args.prompts) > 1

        # magic numbers divined by crowsonkb
        # this normalizes the input batches of cutouts for clip, to prevent overfitting
        # (?and get a better loss?)
        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        # def image_path_to_tensor(path: str, sideX: int, sideY: int) -> Tensor:
        #     img = resize_image(Image.open(path).convert("RGB"), (sideX, sideY))
        #     # Tensor[1, 3, sideX?, sideY?]
        #     return TF.to_tensor(img).unsqueeze(0)

        # def image_tensor_to_embedding(img: Tensor, model: model.CLIP, norm: nn.Module) -> Tensor:
        #     batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        #     # typically [64, 512]
        #     return model.encode_image(norm(batch))

        for path in args.image_prompts:
            is_crossfade = False
            # resize images to the right resolution
            img = resize_image(Image.open(path).convert("RGB"), (sideX, sideY))
            # make some cutouts
            batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
            # embed the image
            embed = self.perceptor.encode_image(normalize(batch)).float()
            # use it as a prompt because prompts just take whatever vectors
            prompts.append(Prompt(embed).to(device))
        # iterations = (
        #     (len(args.prompts) - 1) * (args.dwell + args.fade) + args.dwell
        #     if is_crossfade
        #     else args.max_iterations
        # )

        # uses prompt_queue :(
        @torch.no_grad()
        async def crossfade_prompts(
            prompts: "list[Prompt]", fade: int = 300, dwell: int = 300
        ) -> "list[Prompt]":
            "figure out the prompts and weights to use this iteration"
            if not is_crossfade:
                return prompts
            if prompts[0].dwelt < dwell:
                # we're still dwelling on the current prompt
                prompts[0].dwelt += 1
                # print("dwell: ", prompts[0].dwelt)
            elif prompts[0].weight > 0 and len(prompts) >= 2:
                # there's at least two prompts, and the first one is has nonzero weight
                # we must be fading
                first, second = prompts
                # first prompt gets less weight
                waning_weight = float(first.weight) - 1 / fade
                waxing_weight = min(1.0, float(second.weight) + 1 / fade)
                prompts[0] = Prompt(
                    first.embed, waning_weight, dwelt=first.dwelt, tag=first.tag
                )
                prompts[1] = Prompt(
                    second.embed, waxing_weight, dwelt=second.dwelt, tag=second.tag
                )
            else:
                # first prompt has zero weight, get rid of it
                old_prompt = prompts.pop(0)
                logging.debug("checking next text")
                next_text = ""
                if prompt_queue:
                    next_text = prompt_queue.pop()
                elif redis:
                    maybe = await redis.lpop("stream_queue")
                    if maybe:
                        await redis.rpush("stream_history", maybe)
                        next_text = maybe.decode()
                if next_text:
                    next_prompt = Prompt(
                        self.embed(next_text), weight=0, tag=next_text
                    ).to(device)
                    prompts.append(next_prompt)
                    logging.info("got next text")
                else:
                    old_prompt.weight = torch.as_tensor(1.0)
                    prompts.append(old_prompt)
                    logging.info("no next text, looping")
            if i % args.display_freq == 0:
                logging.info(prompts)
            return prompts

        # uses prompts, prompt_queue...
        async def ascend_txt(i: int, z: Tensor) -> list[Tensor]:
            "synthesize an image and evaluate it for loss"

            logging.debug("await asyncio.to_thread(self.synth(z))")
            out = await asyncio.to_thread(lambda: self.synth(z))
            logging.debug("awaiting synth thread done")

            out = self.synth(z)
            cutouts = make_cutouts(out)
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
                # https://download.pytorch.org/models/vgg16-397923af.pth to /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth
            )
            generated_image_embedding = self.perceptor.encode_image(
                normalize(cutouts)
            ).float()
            if args.video:
                # if we're doing a video we need every single frame
                with torch.no_grad():
                    pil_image = TF.to_pil_image(out[0].cpu())
                    await self.image_queue.put(pil_image)
                    self.frame_times.append(time.time())
                    logging.info(
                        "put image on queue, queue size: %s ", self.image_queue.qsize()
                    )
            losses = []
            # for visualizing cutout transforms:
            # for cutout in cutouts:
            #     loss = prompts[0](self.perceptor.encode_image(normalize(torch.unsqueeze(cutout, 0))))
            #     TF.to_pil_image(cutout).save(f"{loss}.png")

            if args.init_weight:
                losses.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
            for prompt in await crossfade_prompts(prompts, args.fade, args.dwell):
                try:
                    losses.append(prompt(generated_image_embedding))
                except:
                    logging.info(prompt)

            if args.likely:
                assert self.likely_loss
                text_embed = (
                    prompts[0].embed[0] if prompts else torch.zeros(512).to(device)
                )
                massaged = torch.cat(
                    [
                        torch.cat([text_embed, cutout]).unsqueeze(0)
                        for cutout in generated_image_embedding
                    ]
                )
                # break this out into a variable
                losses.append(self.likely_loss(massaged) * 0.7)
            if not losses:
                raise IndexError
            return losses

        writer = None if args.prod else SummaryWriter(comment="vqgan")  # type: ignore

        async def train(i: int) -> float:
            "a single training iteration"
            opt.zero_grad()
            lossAll = await ascend_txt(i, z)
            loss = sum(lossAll)
            if writer:
                writer.add_scalar("loss/train", loss, i)
            if i % args.max_iterations:  # display_freq == 0:
                checkin(i, lossAll)
                # this tells autograd to adjust all the weights based on this new loss

            def opt_go_brr() -> None:
                logging.debug("inside opt_go_brr")
                start = time.time()
                loss.backward()
                opt.step()
                logging.debug("exiting opt_go_brr, elapsed: ", time.time() - start)

            # unleash the GIL
            logging.debug("await asyncio.to_thread(opt_go_brr)")
            await asyncio.to_thread(opt_go_brr)
            logging.debug("awaiting that thread done")

            with torch.no_grad():
                # you have to make sure z isn't too big or too small
                z.copy_(z.maximum(z_min).minimum(z_max))
            return float(loss)

        i = 0
        try:
            while 1:
                # with tqdm() as pbar:
                try:
                    loss = await train(i)
                    await asyncio.sleep(0.001)
                except IndexError as e:
                    traceback.print_exc()
                    break
                if i == args.max_iterations:
                    break
                i += 1
                # pbar.update(1)
        except KeyboardInterrupt:
            pass
        return float(loss), seed
        # steps_without_checkin = 0
        # with tqdm() as pbar:
        #     lossAll = []
        #     for i in range(args.max_iterations):
        #         steps_without_checkin += 1
        #         opt.zero_grad()
        #         return ascend_txt(i)
        #         lossAll = ascend_txt(i)
        #         if i % args.display_freq == 0:
        #             checkin(i, lossAll)
        #             steps_without_checkin = 0
        #         loss = sum(lossAll)
        #         loss.backward()
        #         opt.step()
        #         with torch.no_grad():
        #             z.copy_(z.maximum(z_min).minimum(z_max))
        #         pbar.update(1)
        #     if steps_without_checkin:
        #         checkin(i, lossAll)
        #     return ", ".join(f"{loss.item():g}" for loss in lossAll)


class BetterNamespace:
    "this is just for compatibility with argparse.Namespace, but with updates"

    def __init__(self, **kwargs: Any) -> None:
        self.mapping = kwargs

    def __getattr__(self, attribute: str) -> Any:
        if attribute == "prompts" and "text" in self.mapping:
            return [self.mapping["text"]]
        return self.mapping[attribute]

    def with_update(self, other_dict: "dict[str, Any]") -> "BetterNamespace":
        new_ns = BetterNamespace(**self.mapping)
        new_ns.mapping.update(other_dict)
        return new_ns

    def __repr__(self) -> str:
        return repr(self.mapping)


base_args = BetterNamespace(
    prompts=["dog"],
    image_prompts=[],
    noise_prompt_weights=[],
    size=[780 // 2, 480 // 2],
    init_image=None,
    init_weight=0.0,
    clip_model="ViT-B/32",
    vqgan_config="vqgan_imagenet_f16_16384.yaml",
    vqgan_checkpoint="vqgan_imagenet_f16_16384.ckpt",
    step_size=0.1,
    cutn=64,
    cut_pow=1.0,
    display_freq=25,
    seed=None,
    max_iterations=400,
    fade=100,  # @param {type:"number"}
    dwell=100,  # @param {type: "number"}
    profile=False,  # cprofile
    video=False,
    likely=True,
    prod=True,
    slug=None,
)

if __name__ == "__main__":
    asyncio.run(
        Generator(base_args).generate(
            base_args  # .with_update()#{"max_iterations": 1, "size": [5, 5]})
        )
    )
