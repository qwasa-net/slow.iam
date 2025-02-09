"""
Demo data generator.
"""

from PIL import Image, ImageDraw
import random
import os
import argparse
from math import sin, cos, pi

IMG_WIDTH = 960
IMG_HEIGHT = 960

COLOR_PARTS = [
    "00",
    "33",
    "66",
    "99",
    "cc",
    "ff",
    "aa",
    "55",
    "22",
    "11",
    "44",
    "77",
    "bb",
    "dd",
    "ee",
]


def rnd_color():
    return "#" + "".join(random.choice(COLOR_PARTS) for _ in range(3))


def rnd(b, a=0):
    return min(b, max(a, b * abs(random.normalvariate(0, 1.0))))


def rotate_coords(coords, x, y, a):
    return [
        (
            x + cos(a) * (c[0] - x) - sin(a) * (c[1] - y),
            y + sin(a) * (c[0] - x) + cos(a) * (c[1] - y),
        )
        for c in coords
    ]


def draw_oval(img, x=None, y=None, r=None, color=None):
    color = color or rnd_color()
    x = rnd(img.width) if x is None else x
    y = rnd(img.height) if y is None else y
    r = rnd(min(img.height, img.width) // 3) if r is None else r
    draw = ImageDraw.Draw(img)
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    return img


def draw_rect(img, x=None, y=None, w=None, h=None, a=None, color=None):
    color = color or rnd_color()
    x = rnd(img.width) if x is None else x
    y = rnd(img.height) if y is None else y
    h = rnd(img.height // 2, img.height // 10) if h is None else h
    w = rnd(img.width // 2, img.width // 10) if w is None else w
    a = rnd(2 * pi) if a is None else a
    coords = [(x, y), (x, y + h), (x + w, y + h), (x + w, y)]
    coords = rotate_coords(coords, x + w // 2, y + h // 2, a)
    draw = ImageDraw.Draw(img)
    draw.polygon(coords, fill=color)
    return img


def draw_triangle(img, x=None, y=None, r=None, a=None, color=None):
    color = color or rnd_color()
    x = rnd(img.width) if x is None else x
    y = rnd(img.height) if y is None else y
    r = rnd(min(img.height, img.width) // 4) if r is None else r
    a = rnd(2 * pi) if a is None else a
    coords = [(x, y - r), (x - r, y + r), (x + r, y + r)]
    coords = rotate_coords(coords, x, y, a)
    draw = ImageDraw.Draw(img)
    draw.polygon(coords, fill=color)
    return img


def create_img(w, h, color=None):
    color = color or rnd_color()
    img = Image.new("RGB", (w, h), color)
    return img


def save_image(img, i, args):
    fname = f"{i:04d}.{args.save_format}"
    fpath = os.path.join(args.save_path, fname)
    os.makedirs(args.save_path, exist_ok=True)
    img.save(fpath)
    if args.debug:
        print(f"{fpath}: {img.width}x{img.height} {os.path.getsize(fpath)}")
    return fpath


def draw(img, args):
    if args.figures_count_random:
        fcr = args.figures_count_random
        fc = args.figures_count * (fcr * random.random() + (1 - fcr))
    else:
        fc = args.figures_count
    for i in range(int(fc)):
        if not args.figures or "o" in args.figures:
            img = draw_oval(img)
        if not args.figures or "4" in args.figures:
            img = draw_rect(img)
        if not args.figures or "3" in args.figures:
            img = draw_triangle(img)
    return img


def main():
    args = read_args()
    paths = []
    for i in range(args.count):
        img = create_img(args.img_width, args.img_height, args.bgcolor)
        img = draw(img, args)
        path = save_image(img, i, args)
        paths.append(path)
    print(f"generated {len(paths)} images in {args.save_path}")


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-width", type=int, default=IMG_WIDTH)
    parser.add_argument("--img-height", type=int, default=IMG_HEIGHT)
    parser.add_argument("--bgcolor", type=str, default="white")
    parser.add_argument("--figures", type=str, default="")
    parser.add_argument("--figures-count", type=int, default=25)
    parser.add_argument("--figures-count-random", type=float, default=0.75)
    parser.add_argument("--save-path", type=str, default="data/o34")
    parser.add_argument("--save-format", type=str, default="png")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    if args.debug:
        for k, v in vars(args).items():
            print(f"{k}={v}")

    return args


if __name__ == "__main__":
    main()
