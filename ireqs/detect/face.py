import typing as t

import numpy as np
from insightface.app import FaceAnalysis

from .detect import Detected


class FaceExtractor:
    MODEL_NAME = "buffalo_l"
    DET_TRHRESHOLD = 0.69
    DET_SIZE = (640, 640)

    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.device = config.device
        if "cuda" in str(self.device):
            providers, ctx_id = ["CUDAExecutionProvider"], 0
        else:
            providers, ctx_id = ["CPUExecutionProvider"], -1
        self.app = FaceAnalysis(name=self.MODEL_NAME, providers=providers)
        self.app.prepare(
            ctx_id=ctx_id,
            det_size=self.DET_SIZE,
            det_thresh=self.DET_TRHRESHOLD,
        )

    def detect(self, media_item) -> t.Generator:
        image = media_item.image
        img_np = np.array(image.convert("RGB"))
        faces = self.app.get(img_np)
        if not faces:
            return
        for face in faces:
            box = face.bbox.astype(int)
            cropped = image.crop(box)
            yield Detected(
                media_item=media_item,
                image=cropped,
                box=box,
                score=face.det_score,
                vector=face.embedding,
            )
