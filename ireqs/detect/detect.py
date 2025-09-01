import typing as t
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ihelp.media import MediaItem


class Detector:
    def __init__(self, config, *args, **kwargs):
        self.config = config
        if config.detector == "face":
            from ireqs.detect.face import FaceExtractor  # noqa

            self.detector = FaceExtractor(config)
        elif config.detector in ("feature", "feature-vit-b-32"):
            from ireqs.detect.features import FeaturesExtractorVITB32  # noqa

            self.detector = FeaturesExtractorVITB32(config)
        elif config.detector == "feature-rn101":
            from ireqs.detect.features import FeaturesExtractorRN101  # noqa

            self.detector = FeaturesExtractorRN101(config)
        elif config.detector == "feature-rn50":
            from ireqs.detect.features import FeaturesExtractorRN50  # noqa

            self.detector = FeaturesExtractorRN50(config)
        else:
            raise ValueError(f"Unknown detector type: {config.detector}")

    def detect(self, image) -> t.Generator:
        return self.detector.detect(image)


@dataclass
class Detected:
    media_item: MediaItem | None = None
    image: Image.Image | None = None
    box: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2)
    score: float = 0.0
    vector: np.ndarray | list[float] | None = None
