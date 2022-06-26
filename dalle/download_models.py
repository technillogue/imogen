import wandb
from transformers.modeling_flax_utils import FlaxPreTrainedModel

DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  # high precision
# DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0" # low precision
DALLE_COMMIT_ID = None
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"


# class VQGANPreTrainedModel(FlaxPreTrainedModel):
#     pass

# VQGANPreTrainedModel.from_pretrained(
#     VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
# )


# class PretrainedFromWandbMixin:
#     @classmethod
#     def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
#         """
#         Initializes from a wandb artifact or delegates loading to the superclass.
#         """
#         with tempfile.TemporaryDirectory() as tmp_dir:  # avoid multiple artifact copies
#             if ":" in pretrained_model_name_or_path and not os.path.isdir(
#                 pretrained_model_name_or_path
#             ):
#                 # wandb artifact
#                 if wandb.run is not None:
#                     artifact = wandb.run.use_artifact(pretrained_model_name_or_path)
#                 else:
#                 pretrained_model_name_or_path = artifact.download(tmp_dir)

#             return super(PretrainedFromWandbMixin, cls).from_pretrained(
#                 pretrained_model_name_or_path, *model_args, **kwargs
#             )
#
#
# DalleBart.from_pretrained(
#     DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
# )


def download_wandb(model: str) -> None:
    pretrained_model_name_or_path = model
    artifact = wandb.Api().artifact(pretrained_model_name_or_path)
    artifact.download(".")


download_wandb(DALLE_MODEL)
