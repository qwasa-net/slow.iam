import argparse
import os
import random

from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def random_factor_crop_image(
    img: Image.Image,
    crop_factor: tuple[float, float],
) -> Image.Image:
    w, h = img.size
    crop_w = int(random.uniform(*crop_factor) * w)
    crop_h = int(random.uniform(*crop_factor) * h)
    crop_x = 0 if crop_w >= w else random.randint(0, w - crop_w)
    crop_y = 0 if crop_h >= h else random.randint(0, h - crop_h)
    return img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))


def process_images(
    src: str,
    dst: str,
    min_crop_factor: float = 0.5,
    max_crop_factor: float = 1.0,
    width: int = 384,
    height: int = 384,
    *args,
    **kwargs,
):
    crop_factor = (min_crop_factor, max_crop_factor)
    size = (width, height)
    for root, _, files in os.walk(src):
        for fname in files:
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            src_path = os.path.join(root, fname)
            rel_path = os.path.relpath(src_path, src)
            dst_path = os.path.join(dst, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            try:
                process_image(src_path, dst_path, crop_factor, size)
            except Exception as e:
                print(f"{src_path}: {e}")


def process_image(
    src_path: str,
    dst_path: str,
    crop_factor: tuple[float, float],
    size: tuple[int, int],
):
    img = Image.open(src_path)
    w0, h0 = img.size
    img = random_factor_crop_image(img, crop_factor=crop_factor)
    w1, h1 = img.size
    img = img.resize(size, Image.Resampling.LANCZOS)
    print(f"{src_path}: {w0}x{h0}→{w1}x{h1}→{size[0]}x{size[1]} | {dst_path}")
    img.save(dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--min-crop-factor", type=float, default=0.5)
    parser.add_argument("--max-crop-factor", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    args = parser.parse_args()

    config = vars(args)
    process_images(**config)


if __name__ == "__main__":
    main()
