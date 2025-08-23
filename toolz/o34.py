"""
Demo data generator.
"""

import argparse
import os
import random
import subprocess
from math import cos, pi, sin

from PIL import Image, ImageDraw, ImageFont

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

LIGHT_COLOR_PARTS = [
    "ff",
    "dd",
    "ee",
    "cc",
]


def rnd_color(parts=COLOR_PARTS):
    return "#" + "".join(random.choice(parts) for _ in range(3))


def rnd(b, a=0):
    return min(b, max(a, b * abs(random.normalvariate(0, 1.0))))


def rndp(b, a=0, alpha=2.0, cap=7):
    x = random.paretovariate(alpha)
    x = min(x, cap)
    return a + (b - a) * (x - 1) / (cap - 1)


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


def draw_star(img, x=None, y=None, r=None, a=None, color=None):
    color = color or rnd_color()
    x = rnd(img.width) if x is None else x
    y = rnd(img.height) if y is None else y
    r = rnd(min(img.height, img.width) // 4, img.width // 10) if r is None else r
    a = rnd(2 * pi) if a is None else a
    coords = []
    for i in range(5):
        angle = a + i * 2 * pi / 5
        coords.append((x + r * cos(angle), y + r * sin(angle)))
        angle += pi / 5
        coords.append((x + r / 2 * cos(angle), y + r / 2 * sin(angle)))
    draw = ImageDraw.Draw(img)
    draw.polygon(coords, fill=color)
    return img


def draw_blot(img, x=None, y=None, r=None, color=None, n=33, distpw=0.25):
    color = color or rnd_color()
    x = rnd(img.width) if x is None else x
    y = rnd(img.height) if y is None else y
    r = rnd(min(img.height, img.width) // 3, img.width // 10) if r is None else r
    draw = ImageDraw.Draw(img)
    coords = []
    distp = r
    for i in range(n):
        angle = i * 2 * pi / n + rnd(pi / (n * 3), -pi / (n * 3))
        d = rndp(r, r * 0.25, alpha=2, cap=10.0)
        dist = d * (1 - distpw) + distp * distpw
        distp = dist
        px = x + dist * cos(angle)
        py = y + dist * sin(angle)
        coords.append((px, py))
    draw.polygon(coords, fill=color)

    return img


def draw_txt(img, text: str, h=None, color=None, font_names="roboto,arial,menlo,sans"):
    color = color or rnd_color()
    h = h or 0.8 * img.height
    try:
        font_path = get_font_path(font_names)
        font = ImageFont.truetype(font_path, h)
    except Exception:  # noqa
        font = ImageFont.load_default(h)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font, spacing=0)
    ascent, descent = font.getmetrics()
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1] + (ascent + descent) // 2
    tx, ty = (img.width - tw) // 2, (img.height - th) // 2
    draw.text((tx, ty), text, fill=color, font=font)
    return img


def create_img(w, h, color=None):
    color = color or rnd_color()
    return Image.new("RGB", (w, h), color)


def save_image(img, i, args):
    if args.save_name and args.count > 1:
        fname = f"{args.save_name}-{i:04d}.{args.save_format}"
    elif args.save_name:
        fname = f"{args.save_name}.{args.save_format}"
    else:
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
    for _ in range(int(fc)):
        if not args.figures or "o" in args.figures:
            img = draw_oval(img)
        if not args.figures or "4" in args.figures:
            img = draw_rect(img)
        if not args.figures or "3" in args.figures:
            img = draw_triangle(img)
        if not args.figures or "5" in args.figures:
            img = draw_star(img)
        if "b" in args.figures:
            img = draw_blot(img)
    if args.text:
        img = draw_txt(img, args.text, font_names=args.text_font)
    return img


def main():
    args = read_args()
    paths = []
    for i in range(args.count):
        bgcolor = rnd_color(LIGHT_COLOR_PARTS) if args.bgcolor == "random" else args.bgcolor
        img = create_img(args.img_width, args.img_height, bgcolor)
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
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text-font", type=str, default="roboto,arial,menlo,sans")
    parser.add_argument("--save-path", type=str, default="data/o34")
    parser.add_argument("--save-name", type=str, default="")
    parser.add_argument("--save-format", type=str, default="png")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    if args.debug:
        for k, v in vars(args).items():
            print(f"{k}={v}")

    return args


def get_font_path(font_names):
    for name in font_names.split(","):
        name = name.strip()
        try:
            output = subprocess.check_output(["fc-match", "-f", "%{file}\n", name])
            font_path = output.strip()
            if os.path.isfile(font_path):
                return font_path
        except Exception:  # noqa
            pass
    return None


if __name__ == "__main__":
    main()
