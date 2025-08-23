import math
import random

from PIL import Image
from torchvision.transforms import functional as trf
from torchvision.transforms import v2 as trv2


class RandomResizedCropSq(trv2.RandomResizedCrop):
    def make_params(self, flat_inputs):
        height, width = trv2.query_size(flat_inputs)
        if height == width:
            return dict(top=0, left=0, height=height, width=width)
        return super().make_params(flat_inputs)

    def transform(self, inpt, params):
        return super().transform(inpt, params)


class RandomRotateChoice:
    """
    RandomRotateChoice is a transform
    that rotates an image by a random angle.
    """

    def __init__(self, angles: list):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        if angle == 0:
            return x
        return trf.rotate(x, angle, expand=True)


class FixedAspectResize:
    """
    FixedAspectResize is a transform
    that resizes an image to a fixed aspect ratio.
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.h, self.w = size, size
        elif isinstance(size, tuple) and len(size) == 2:
            self.h, self.w = size
        else:
            raise ValueError("size must be int or tuple (w,h)")
        self.ratio = self.w / self.h

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError(f"img must be PIL.Image, not {type(img)}")
        iw, ih = img.size
        if iw == self.w and ih == self.h:
            return img
        iratio = float(iw) / float(ih)
        if iratio > self.ratio:
            w, h = math.ceil(self.h * iratio), self.h
        else:
            w, h = self.w, math.ceil(self.w / iratio)
        return trf.resize(img, (h, w), interpolation=Image.BILINEAR)


class SuperCrop:
    """
    SuperCrop is a transform that crops an image
    to a specified size and applies additional transformations.
    """

    def __init__(self, size_crop, size_resize):
        self.transformers = [
            trv2.Resize((size_resize, size_resize)),
            trv2.Compose([trv2.Resize(size_resize), trv2.CenterCrop(size_crop)]),
            trv2.Compose([trv2.Resize((size_resize, size_resize)), self._rot180]),
            trv2.Compose([trv2.Resize(size_resize), self._rot90, trv2.CenterCrop(size_crop)]),
            trv2.Compose([trv2.Resize(int(size_resize * 1.23)), trv2.CenterCrop(size_crop)]),
        ]

    def _gray(self, img):
        imgg = trv2.functional.adjust_saturation(img, 0.02)
        return trv2.functional.adjust_contrast(imgg, 1.2)

    def _rot90(self, img):
        return trv2.functional.rotate(img, angle=90)

    def _rot270(self, img):
        return trv2.functional.rotate(img, angle=270)

    def _rot180(self, img):
        return trv2.functional.rotate(img, angle=180)

    def __call__(self, img):
        return [t(img) for t in self.transformers]


class SuperMiniCrop(SuperCrop):
    """
    SuperMiniCrop is a transform that crops an image
    to a specified size and applies additional transformations.
    """

    def __init__(self, size_crop, size_resize):
        self.transformers = [
            trv2.Resize((size_resize, size_resize)),
            trv2.Compose([trv2.Resize(int(size_resize)), trv2.CenterCrop(size_crop)]),
        ]


class AnyTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        if isinstance(x, (list, tuple)):
            return type(x)(self.transform(item) for item in x)
        return self.transform(x)
