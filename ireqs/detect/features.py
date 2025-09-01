import typing as t

import open_clip
import torch
from PIL import Image, ImageOps

from .detect import Detected


class FeaturesExtractor:
    MODEL_NAME = None
    PRETRAINED = None
    VECTOR_SIZE = 512

    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.device = config.device
        self.projection = None

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.MODEL_NAME,
            pretrained=self.PRETRAINED,
            device=self.device,
            force_quick_gelu=True,
        )
        self.model.eval()

    @torch.no_grad()
    def detect(self, media_item) -> t.Generator[Detected, None, None]:
        image = Image.open(media_item.path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor).squeeze(0)

        if self.projection:
            features = self.projection(features)

        yield Detected(
            media_item=media_item,
            image=None,
            box=(0, 0, image.width, image.height),
            vector=features.cpu().numpy(),
        )


class FeaturesExtractorRN101(FeaturesExtractor):
    MODEL_NAME = "RN101"
    PRETRAINED = "openai"


class FeaturesExtractorRN50(FeaturesExtractor):
    MODEL_NAME = "RN50"
    PRETRAINED = "openai"
    VECTOR_SIZE = 1024

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.projection = torch.nn.Linear(1024, 512).to(self.device)


class FeaturesExtractorVITB32(FeaturesExtractor):
    MODEL_NAME = "ViT-B-32"
    PRETRAINED = "openai"


class FeaturesExtractorVITB16(FeaturesExtractor):
    MODEL_NAME = "ViT-B-16"
    PRETRAINED = "openai"
