import logging
import math
import os
import random
import sys

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as tvf
from torchvision.transforms import v2 as tvt2


class DatasetTransform(Dataset):
    """
    DatasetTransform is a wrapper for a dataset
    that applies a transform to each item.
    """

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.samples = dataset.samples

    def __getitem__(self, index):
        target = self.dataset[index][1]
        if self.transform:
            item = self.transform(self.dataset[index][0])
        else:
            item = self.dataset[index][0]
        return item, target

    def __len__(self):
        return len(self.dataset)


class MultiLabelDataset(Dataset):
    """
    MultiLabelDataset is a wrapper for a dataset
    that converts a path to a multi-label target.
    """

    def __init__(self, dataset, classes=None, drop_prefix=None):
        self.dataset = dataset
        if not classes or not isinstance(classes, list):
            self.classes = dataset.classes
            self.class_to_idx = dataset.class_to_idx
        else:
            self.classes = classes
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = dataset.samples
        self.drop_prefix = drop_prefix
        self._cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        targets = self.path_to_targets(self.dataset.samples[index][0])
        return img, targets

    def path_to_targets(self, path):
        if self.drop_prefix:
            path = path[len(self.drop_prefix) :]
        dirname, filename = os.path.split(path)
        if dirname in self._cache:
            return self._cache[dirname]
        path_parts = set(filter(lambda x: not str(x).startswith("_"), dirname.split(os.sep)))
        path_parts.intersection_update(self.classes)
        path_targets = [c in path_parts for c in self.class_to_idx]
        # if sum(path_targets) > 2:
        #     print(f"{path} -> {path_parts} → {''.join(map(lambda x: str(int(x)), path_targets))}")
        tt = torch.tensor(path_targets, dtype=torch.bool)
        self._cache[dirname] = tt
        return tt


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
        return tvf.rotate(x, angle, expand=True)


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
        # print(f"resize {iw}x{ih} ({iratio}) → {w}x{h} ({self.ratio}/{w/h})")
        return tvf.resize(img, (h, w), interpolation=Image.BILINEAR)


class SuperCrop:
    """
    SuperCrop is a transform that crops an image
    to a specified size and applies additional transformations.
    """

    def __init__(self, size_crop, size_resize):
        self.transformers = [
            tvt2.Resize((size_resize, size_resize)),
            tvt2.Compose([tvt2.Resize((size_resize, size_resize)), self._rot180]),
            tvt2.Compose([tvt2.Resize(size_resize), self._rot90, tvt2.CenterCrop(size_crop)]),
            tvt2.Compose([tvt2.Resize(int(size_resize * 1.23)), tvt2.CenterCrop(size_crop)]),
            tvt2.Compose([tvt2.Resize(size_resize), tvt2.CenterCrop(size_crop)]),
        ]

    def _gray(self, img):
        imgg = tvt2.functional.adjust_saturation(img, 0.02)
        return tvt2.functional.adjust_contrast(imgg, 1.2)

    def _rot90(self, img):
        return tvt2.functional.rotate(img, angle=90)

    def _rot270(self, img):
        return tvt2.functional.rotate(img, angle=270)

    def _rot180(self, img):
        return tvt2.functional.rotate(img, angle=180)

    def __call__(self, img):
        return [t(img) for t in self.transformers]


class SuperMiniCrop(SuperCrop):
    """
    SuperMiniCrop is a transform that crops an image
    to a specified size and applies additional transformations.
    """

    def __init__(self, size_crop, size_resize):
        self.transformers = [
            tvt2.Resize((size_resize, size_resize)),
            tvt2.Compose([tvt2.Resize((size_resize, size_resize)), self._rot180]),
            tvt2.Compose([tvt2.Resize(int(size_resize * 1.1)), tvt2.CenterCrop(size_crop)]),
        ]


class Keeper:
    """
    Keeper is a naïve key-value storage (dict wrapper).
    """

    data = {}
    keep = True

    def put(self, key, value):
        if self.keep:
            self.data[key] = value

    def get(self, key):
        value = self.data.get(key)
        return value

    def pop(self, key):
        value = self.get(key)
        del self.data[key]
        return value


class AccumulatorIterator:
    """
    AccumulatorIterator is an iterator that stores results.
    On the first iteration, it collects results from the source iterator.
    On subsequent iterations, it returns the stored results.
    """

    def __init__(self, source):
        self.data = []
        self.data_position = 0
        self.source = source
        self.source_oef = False

    def __iter__(self):
        self.data_position = 0
        return self

    def __next__(self):
        if self.data_position < len(self.data):
            item = self.data[self.data_position]
            self.data_position += 1
            return item
        if self.source_oef:
            raise StopIteration
        try:
            item = next(self.source)
            self.data.append(item)
            self.data_position = len(self.data)
            return item
        except StopIteration:
            self.source_oef = True
            raise StopIteration


def str_eclipse(s, limit=80, r=0.75):
    if not s or len(s) <= limit:
        return s
    p = int(limit * r)
    return s[:p] + "…" + s[-(limit - p - 1) :]


def sizxx(sz):
    if isinstance(sz, torch.Tensor):
        return "x".join(map(str, sz.size()))
    if isinstance(sz, torch.Size):
        return "x".join(map(str, sz))
    if isinstance(sz, (int, float)):
        return str(sz)
    if isinstance(sz, (tuple, list)):
        return "x".join(map(str, sz))
    return "-"


def setup_logging(args):
    """
    Configures logging.
    """

    level = getattr(logging, args.log_level, logging.INFO)
    fmt = "%(asctime)s %(message)s"
    datefmt = "[%Y-%m-%d %H:%M:%S]"

    handlers = [
        logging.StreamHandler(sys.stderr),
    ]

    if args.log:
        handlers.append(logging.FileHandler(args.log))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
