import argparse
import io
import os
import time
import urllib.request

from duckduckgo_search import DDGS
from PIL import Image


def ddg_images_search(query, max_results=100):
    ddgs = DDGS()
    results = ddgs.images(query, safesearch="off", max_results=max_results)
    return [r["image"] for r in results]


def save_images(images, path):
    i = 0
    os.makedirs(path, exist_ok=True)
    for image in images:
        try:
            fname = os.path.basename(image)
            ipath = os.path.join(path, fname)
            while os.path.exists(ipath):
                ipath = os.path.join(path, f"{int(time.time())}-{fname}")
            with urllib.request.urlopen(image, timeout=3) as f:
                data = f.read()
                img = Image.open(io.BytesIO(data))
                print(img.format, img.width, img.height)
                if img.format not in ["JPEG", "PNG"]:
                    print(f"Skip bad format {fname} {img.format}")
                    continue
                if img.width < 500 or img.height < 500:
                    print(f"Skip too small {fname} {img.width}x{img.height}")
                    continue
                open(ipath, "wb").write(data)  # noqa
            i += 1
            print(f"+{ipath}")
        except Exception as x:
            print(x)
    return i


def main():
    args = read_args()
    for query in args.query:
        images = ddg_images_search(query, max_results=args.max_results)
        print(f"Found {len(images)} images for {query}")

        path = os.path.join(args.save_path, query)
        i = save_images(images, path)
        print(f"Saved {i} images to {path}")


def read_args():
    parser = argparse.ArgumentParser(description="Search images on DuckDuckGo.")
    parser.add_argument("query", type=str, default="cat", nargs="+")
    parser.add_argument("--save-path", type=str, default="./images/")
    parser.add_argument("--max-results", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    main()
