from dataclasses import dataclass
from clip import clip, model
from torch import Tensor, nn
from typing import NewType, TypeVar

def embed(perceptor: model.CLIP, text: str) -> Tensor:
    "normalized clip embedding of text"
    device = perceptor.token_embedding.weight.device
    tokens = clip.tokenize(text, truncate=True).to(device)
    embedding = perceptor.encode_text(tokens)  # tensor[1, 512]
    normed_embedding = nn.functional.normalize(embedding, dim=1)
    return normed_embedding


@dataclass
class BasicPrompt:
    prompt: str
    reacts: int
    loss: float

    def __post_init__(self) -> None:
        self.label = float(bool(self.reacts))


@dataclass
class FilePrompt(BasicPrompt):
    filepath: str

    def __post_init__(self) -> None:
        self.slug = (
            self.filepath.removeprefix("output/")
            .removesuffix(".png")
            .removesuffix("/progress")
        ) + ".png"


@dataclass
class Prompt(BasicPrompt):
    embed: Tensor


@dataclass
class TokenPrompt(BasicPrompt):
    tokens: Tensor


@dataclass
class ImgPrompt(Prompt):
    image_embed: Tensor


BonelessPrompt = TypeVar("BonelessPrompt", BasicPrompt, FilePrompt)
EmbedPrompt = TypeVar("EmbedPrompt", Prompt, ImgPrompt)
