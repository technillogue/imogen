#!python3
import argparse
import logging
import random
import sys
from datetime import datetime
from types import SimpleNamespace
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from dalle_mini import DalleBart, DalleBartProcessor
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
from PIL import Image
from vqgan_jax.modeling_flax_vqgan import VQModel

#DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  # high precision
DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0" # low precision
DALLE_COMMIT_ID = None
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

start = datetime.now()


class Dalle:
    def __init__(self) -> None:
        jax.local_device_count()
        logging.info(f"{datetime.now() - start} Loading Dalle model ...")
        self.model, params = DalleBart.from_pretrained(
            DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
        )
        self.params = replicate(params)
        logging.info("done.")

        logging.info(f"{datetime.now() - start} Loading VQGAN model ...")
        self.vqgan, vqgan_params = VQModel.from_pretrained(
            VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
        )
        self.vqgan_params = replicate(vqgan_params)
        logging.info("done.")

        logging.info(f"{datetime.now() - start} Processing inputs ... ")
        self.processor = DalleBartProcessor.from_pretrained(
            DALLE_MODEL, revision=DALLE_COMMIT_ID
        )

    def generate(self, args: argparse.Namespace) -> str:
        tokens = replicate(self.processor([args.inputs]))

        # Parameters: https://huggingface.co/blog/how-to-generate).
        gen_top_k = None
        gen_top_p = None
        temperature = None
        cond_scale = 10.0

        @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
        def p_generate(
            tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
        ):
            return self.model.generate(
                **tokenized_prompt,
                prng_key=key,
                params=params,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                condition_scale=condition_scale,
            )

        @partial(jax.pmap, axis_name="batch")
        def p_decode(indices, params):
            return self.vqgan.decode_code(indices, params=params)

        seed = random.randint(0, 2**32 - 1)
        key = jax.random.PRNGKey(seed)
        _, subkey = jax.random.split(key)
        shared_key = shard_prng_key(subkey)
        encoded_images = p_generate(
            tokens,
            shared_key,
            self.params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )
        logging.info("done.")

        logging.info(f"{datetime.now() - start} Encoding images ... ")
        encoded_images = encoded_images.sequences[..., 1:]  # remove BOS
        logging.info("done.")

        logging.info(f"{datetime.now() - start} Decoding images ... ")

        decoded_images = p_decode(encoded_images, self.vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        logging.info("done.")

        logging.info(f"{datetime.now() - start} Showing images ... ")
        for decoded_img in decoded_images:
            img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
            img.save(args.path)
        logging.info(f"done.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        Dalle().generate(SimpleNamespace(inputs=" ".join(sys.argv[1:]), path="img.png"))
    else:
        DalleBart.from_pretrained(
            DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
        )
        VQModel.from_pretrained(
            VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
        )

