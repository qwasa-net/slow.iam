import math
import random

from PIL import Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset


def str_eclipse(s, limit=80, r=0.75):
    if not s or len(s) <= limit:
        return s
    p = int(limit * r)
    return s[:p] + "…" + s[-(limit - p - 1) :]


class DatasetTransform(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        target = self.dataset[index][1]
        if self.transform:
            item = self.transform(self.dataset[index][0])
        else:
            item = self.dataset[index][0]
        return item, target

    def __len__(self):
        return len(self.dataset)


class RandomRotateChoice:
    def __init__(self, angles: list):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)


class FixedAspectResize:
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
        # print(f"resize {iw}x{ih} ({iratio}) → {w}x{h} ({self.ratio}/{w/h})")
        return F.resize(img, (h, w), interpolation=Image.BILINEAR)
