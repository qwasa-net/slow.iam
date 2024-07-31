"""
"""

import tkinter as tk
from PIL import Image, ImageTk
import argparse
import random
import os
import re
import shutil

IMG_SIZE = 1024


def show_image(img, title="Image"):
    print(f"showing image {title}")
    image = img.resize((1920, 1080))
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(image=photo)
    label.image = photo
    label.pack()


def load_image(image_path):
    try:
        img = Image.open(image_path)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def get_images(args):

    img_paths = []
    path_filter = re.compile(args.data_filter) if args.data_filter else None
    if os.path.isdir(args.data):
        for root, dirs, files in os.walk(args.data):
            for file in files:
                img_path = os.path.join(root, file)
                if path_filter and not path_filter.search(img_path, re.I):
                    continue
                img_paths.append(img_path)
    elif os.path.isfile(args.data):
        img_paths.append(args.data)

    if args.limit:
        random.shuffle(img_paths)
        img_paths = img_paths[: args.limit]
    else:
        img_paths.sort()

    for ipath in img_paths:
        img = load_image(ipath)
        if img is not None:
            yield img, ipath


class QC:

    image = None
    root = None
    img = None
    txt = None
    textvariable = None
    state: int = 0
    xy0 = (0, 0)
    xy1 = (0, 0)

    def __init__(self, root, images, args):
        self.images = images
        self.args = args
        self.build_ui(root)
        self.reset_selection()

    def build_ui(self, root):
        self.root = root
        self.img = tk.Label(root)
        self.img.pack(padx=10, pady=10)
        self.textvariable = tk.StringVar()
        self.txt = tk.Label(root, textvariable=self.textvariable, font=("Sans", 10))
        self.textvariable.set("Hello")
        self.txt.pack(padx=10, pady=10)
        self.root.bind_all("<Button>", lambda x: self.click_handler(x))

    def click_handler(self, event):
        print(f"clicked at {event.x}, {event.y} {event.num}")
        if event.num == 3:
            self.reset_selection()
            self.show_image()
        elif event.num != 1:
            return
        elif self.state == 0:
            self.show_image()
        elif self.state == 1:
            self.xy0 = (event.x, event.y)
            self.state = 2
        elif self.state == 2:
            self.xy1 = (event.x, event.y)
            if self.xy1[0] < self.xy0[0] or self.xy1[1] < self.xy0[1]:
                self.reset_selection()
                self.state = 1
            else:
                self.crop_image()
                self.state = 3
        elif self.state == 3:
            self.state = 0
            self.save_image()
            self.reset_selection()

        if self.state == 4:
            text = "[END]"
        else:
            iname = os.path.basename(self.ipath) if self.ipath else ""
            text = (
                f"state={self.state} "
                f"{iname=} "
                f"(x0,y0)={self.xy0} (x1,y1)={self.xy1} "
                f"(w,h)={(self.xy1[0]-self.xy0[0], self.xy1[1]-self.xy0[1])}"
            )
        self.textvariable.set(text)

    def reset_selection(self):
        self.xy0 = (0, 0)
        self.xy1 = (0, 0)

    def save_image(self):
        print(f"save image {self.ipath}")
        if self.args.bak:
            shutil.copyfile(self.ipath, self.ipath + ".bak")
        self.image.save(self.ipath)

    def crop_image(self):
        self.image = self.image.crop((*self.xy0, *self.xy1))
        photo = ImageTk.PhotoImage(self.image)
        self.img.configure(image=photo)
        self.img.image = photo

    def show_image(self):
        try:
            image, ipath = next(self.images)
        except StopIteration:
            print("No more images")
            self.state = 4
            return
        ih, iw = image.height, image.width
        h, w = self.root.winfo_height(), self.root.winfo_width()
        h = int(h * 0.9)
        w = int(h / ih * iw)
        self.image = image.resize((w, h))
        photo = ImageTk.PhotoImage(self.image)
        self.img.configure(image=photo)
        self.img.image = photo
        self.ipath = ipath
        self.state = 1


def main():
    args = read_args()

    images = get_images(args)
    root = tk.Tk()
    root.geometry(f"{args.win_width}x{args.win_height}")
    qc = QC(root, images, args)
    root.bind("<Escape>", lambda e: root.quit())
    root.mainloop()


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-width", type=int, default=IMG_SIZE)
    parser.add_argument("--img-height", type=int, default=IMG_SIZE)
    parser.add_argument("--win-width", type=int, default=2600)
    parser.add_argument("--win-height", type=int, default=1600)
    parser.add_argument("--data", type=str, default="data/")
    parser.add_argument("--data-filter", type=str, default="(jpg|png|jpeg)$")
    parser.add_argument("--bak", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}={v}")

    return args


if __name__ == "__main__":
    main()
