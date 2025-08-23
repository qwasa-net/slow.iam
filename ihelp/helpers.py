import gzip
import logging
import os
import sys

import torch
from PIL import Image
from torchvision.transforms import functional as trf

log = logging.getLogger()


def tenzor_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")
        return trf.to_tensor(img)


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
        return self.data.get(key)

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
            raise StopIteration from None


def str_eclipse(s, limit=80, r=0.75):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    if not s or len(s) <= limit:
        return s
    p = int(limit * r)
    return s[:p] + "…" + s[-(limit - p - 1) :]


def sizexx(sz):
    if isinstance(sz, torch.Tensor):
        return "x".join(map(str, sz.size()))
    if isinstance(sz, torch.Size):
        return "x".join(map(str, sz))
    if isinstance(sz, (int, float)):
        return str(sz)
    if isinstance(sz, (tuple, list)):
        return str(len(sz))
    if hasattr(sz, "shape"):
        return "x".join(map(str, sz.shape))
    return "-"


def setup_logging(args):
    """
    Configures logging.
    """

    level = getattr(logging, getattr(args, "log_level", None), logging.INFO)

    fmt = "%(asctime)s %(message)s"
    datefmt = "[%y-%m-%d %H:%M:%S]"

    handlers = [
        logging.StreamHandler(sys.stderr),
    ]

    if getattr(args, "log", None):
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


def if_debug(logg=log):
    return logg.getEffectiveLevel() <= logging.DEBUG


def log_tz(prefix="", **kwargs):
    if not if_debug(log):
        return
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            log.debug("%s%s: std=%.3f mean=%.3f [%.3f, %.3f]", prefix, k, v.std(), v.mean(), v.min(), v.max())
        else:
            log.debug("%s%s: %s", prefix, k, str_eclipse(str(v), 80, 0.25))
