import kornia.augmentation as K
import torch
from clip import clip, model
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms
from torchvision.transforms import functional as TF
from utils import resample


class ImageEmbedder:
    def __init__(self, perceptor: model.CLIP, args: dict) -> None:
        self.perceptor = perceptor
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode="border"), # type: ignore
            K.RandomPerspective(0.2, p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
        )
        self.cut_size = self.perceptor.visual.input_resolution
        self.cutn = args["cutn"]
        self.cut_pow = args["cut_pow"]

    def embed(self, path: str) -> Tensor:
        img = Image.open(path).convert("RGB")  # resize_image(_, (sideX, sideY))
        input = (
            TF.to_tensor(img)
            .unsqueeze(0)
            .to(self.perceptor.token_embedding.weight.device)
        )
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
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size] # type: ignore
            # resample to CLIP's input resolution
            resampled = resample(cutout, (self.cut_size, self.cut_size))
            cutouts.append(resampled)
        # run each cutout through those augmentations
        auged_batch = self.augs(torch.cat(cutouts, dim=0))
        image_embedding = self.perceptor.encode_image(
            self.normalize(auged_batch)
        ).float()
        return image_embedding
