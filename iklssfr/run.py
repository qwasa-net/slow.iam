import argparse
import json
import logging
import math
import os
import sys
import urllib.parse
from collections import Counter
from itertools import batched

import torch
from torchvision.transforms import v2

import ihelp.media
from ihelp.config import Configuration
from ihelp.helpers import AccumulatorIterator, Keeper, log_tz, setup_logging, str_eclipse

from .transforms import AnyTransform, SuperCrop, SuperMiniCrop

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

PRED_MIN = 0.01
PRED_DEV = 3.45

LOG_FORMAT = "%(message)s"  # "%(asctime)s %(levelname)s %(message)s"

log = logging.getLogger()
orig_keeper = Keeper()


@torch.no_grad()
def classify(model, dataloader, classes, args):
    """
    Classify images from dataloader using model and classes.
    """
    model.eval()
    torch.set_grad_enabled(False)

    # + every source image could be split into parts (crops, rotations)
    # + each part to be classified independently
    # + all part predictions are combined into one
    for i, (img_group, label) in enumerate(dataloader, start=1):
        # tupalize single tensor
        if not isinstance(img_group, (tuple, list)):
            img_group = (img_group,)

        # keep here every part prediction
        outs = []

        for j, img in enumerate(img_group, start=1):
            img = img.to(args.device)
            out = model(img)[0]
            outs.append(list(out))
            log_tz(f"#{i}+{j}: ", img=img, out=out)
            if len(img_group) > 1:
                rz = ClassifierResultItem(label, img, out, classes, part=j, no=i)
                log.debug(f"#{i}+{j}: {rz}")
                if args.show_parts:
                    yield rz

        # combine all part predictions into one
        rz = ClassifierResultItem(label, img, sqeeze_outs(outs, args), classes, no=i)
        log.info(f"#{i}: {rz}")

        # send current result to the caller
        yield rz


class ClassifierResultItem:
    def __init__(self, label, data, preds, classes, part=None, no=0):
        assert len(preds) == len(classes), "preds and classes must have the same length"
        assert isinstance(label, str), "label must be a string"
        assert isinstance(preds, (list, torch.Tensor)), "preds must be a list"
        assert isinstance(classes, dict), "classes must be a dict"
        self.data = data
        self.label = label
        self.preds = list(preds) if isinstance(preds, torch.Tensor) else preds
        self.classes = classes
        self.is_part = part
        self.no = no

        preds_sorted = sorted(enumerate(self.preds), key=lambda x: -x[1])
        self.pred_ids = {x[0]: float(x[1]) for x in preds_sorted}
        self.pred_names = {classes.get(p, p): w for p, w in self.pred_ids.items()}

    def __str__(self):
        lbl = str_eclipse(os.path.basename(self.label), 40)
        prds = {k: round(v, 2) for k, v in self.pred_names.items()}
        return f"[{lbl}] {prds}"

    def confident_predictions(self):
        preds_pmin = list(filter(lambda x: x[1] > PRED_MIN, self.pred_names.items()))
        if not preds_pmin:
            return ""
        selected = [preds_pmin[0]]
        for n, w in preds_pmin[1:]:
            if w > PRED_MIN and w > selected[0][1] / PRED_DEV:
                selected.append((n, w))
        return ",".join([n for (n, w) in selected[:100]])


def sqeeze_outs(combined, args):
    """
    Squeeze multiple outputs into one by taking max and mean
    and median and sum and max and mean and max and mean and sum
    to get the best combined prediction.
    """
    assert isinstance(combined, (list, torch.Tensor)), "combined must be a list"
    assert len(combined) > 0, "combined must not be empty"
    assert isinstance(combined[0], (list, torch.Tensor)), "combined must be a list of lists"

    if len(combined) == 1:
        return [float(c) for c in combined[0]]

    # normalize confident abssence
    combined = [[float(c) if c >= args.predict_min else 0.0 for c in cs] for cs in combined]

    # mean
    combined_mean = [sum(cs) / len(combined) for cs in zip(*combined, strict=False)]

    # get detected
    det_count = [0] * len(combined[0])
    det_mean = [0] * len(combined[0])
    for cc in combined:
        for i, c in enumerate(cc):
            if c >= args.predict_min:
                det_count[i] += 1
                det_mean[i] += c
    det_confidence = len(combined) / PRED_DEV
    det_count = [dc if dc > det_confidence else len(combined) for dc in det_count]
    det_mean = [x / det_count[i] if det_count[i] > 0 else 0.0 for i, x in enumerate(det_mean)]

    # use max, dampened
    max_dmpd = [max(cs) / len(combined) for cs in zip(*combined, strict=False)]

    # combine average detected confidence and the damped max (magic!)
    combined = [max(combined_mean[j], det_mean[j]) + max_dmpd[j] for j in range(len(combined[0]))]
    # log.debug(f"{combined_mean=} / {det_mean=} / {max_dmpd=} / {combined=}")

    return combined


def get_dataloader(config):
    transforms = get_transforms(config)
    for src in config.src:
        loader = ihelp.media.load_data(src, config)
        for mi in loader:
            img = mi.image
            orig_keeper.put(mi.path, img)
            if transforms:
                img = transforms(img)
            if isinstance(img, torch.Tensor):
                img.unsqueeze_(0)
            yield img, mi.path


def get_transforms(config):
    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    if config.data_autocontrast:
        transforms.append(v2.functional.autocontrast)
    if config.data_equalize:
        transforms.append(v2.functional.equalize)
    if config.data_blur:
        transforms.append(v2.GaussianBlur(kernel_size=(3, 7), sigma=0.5))
    if config.data_crop in ("5", "5+"):
        transforms.append(v2.Resize(size=int(config.size_resize * 1.33)))
        transforms.append(v2.FiveCrop(size=config.size_crop))
        transforms.append(v2.Lambda(lambda c: c.unsqueeze(0)))
    elif config.data_crop in ("10", "10+"):
        transforms.append(v2.Resize(size=int(config.size_resize * 1.33)))
        transforms.append(v2.TenCrop(size=config.size_crop))
        transforms.append(v2.Lambda(lambda c: c.unsqueeze(0)))
    elif config.data_crop in ("s", "super"):
        transforms.append(SuperCrop(config.size_crop, config.size_resize))
        transforms.append(v2.Lambda(lambda c: c.unsqueeze(0)))
    elif config.data_crop in ("sm", "supermini"):
        transforms.append(SuperMiniCrop(config.size_crop, config.size_resize))
        transforms.append(v2.Lambda(lambda c: c.unsqueeze(0)))
    elif config.data_crop in ("c", "center"):
        transforms.append(v2.Resize(size=config.size_resize))
        transforms.append(v2.CenterCrop(size=config.size_crop))
    else:
        transforms.append(v2.Resize(size=(config.size_resize, config.size_resize)))
        transforms.append(v2.CenterCrop(size=config.size_crop))
    if config.data_normalize:
        transforms.append(AnyTransform(v2.Normalize(IMG_MEAN, IMG_STD)))
    return v2.Compose(transforms)


def main():
    config = read_args()
    setup_logging(config)

    log.info("config:")
    for k, v in config.items():
        log.info(f"{k}={v}")

    config, model, classes = load_model(config)
    model.to(config.device)

    orig_keeper.keep = config.show_original

    dataloader = get_dataloader(config)
    class_results = classify(model, dataloader, classes, config)
    result = AccumulatorIterator(class_results)

    has_video = config.data_is_videostream or any(
        [str_eclipse(p).lower().endswith(tuple(ihelp.media.VIDEO_EXTENSIONS)) for p in config.src]
    )

    if config.show:
        show(
            result,
            config=config,
            pagesize=36 if has_video else 64,
        )

    if config.summarize:
        summarize(result, config)

    if config.save:
        save(result, config)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "src",
        type=str,
        nargs="+",
        default="data/",
    )
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
    )
    parser.add_argument(
        "--save",
        "-o",
        type=str,
    )
    parser.add_argument(
        "--data-is-videostream",
        "-dv",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--data-video-step",
        "-dvs",
        type=int,
        default=ihelp.media.VIDEO_STEP_SEC,
    )
    parser.add_argument(
        "--data-video-limit",
        "-dvl",
        type=int,
        default=ihelp.media.VIDEO_CAPS_LIMIT,
    )
    parser.add_argument(
        "--data-normalize",
        "-dn",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--data-equalize",
        "-dq",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--data-blur",
        "-db",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--data-autocontrast",
        "-da",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--data-crop",
        "-dc",
        type=str,
        default="",
        choices=["", "c", "center", "5", "10", "s", "super", "sm"],
    )
    parser.add_argument(
        "--size-crop",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--size-resize",
        "-z",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--show",
        "-s",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--show-original",
        "-so",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--show-parts",
        "-sp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--show-original-squeeze",
        action=argparse.BooleanOptionalAction,
        default=True,  # 8)
    )
    parser.add_argument(
        "--predict-min",
        "-pm",
        type=float,
        default=PRED_MIN,
    )
    parser.add_argument(
        "--predict-dev",
        "-pd",
        type=float,
        default=PRED_DEV,
    )
    parser.add_argument(
        "--summarize",
        "-ss",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    if args.show_original:
        args.show = True

    if args.size_resize and (args.size_crop > args.size_resize or args.size_crop == 0):
        args.size_crop = args.size_resize

    args.src = [urllib.parse.unquote(s[7:]) if s.startswith("file://") else s for s in args.src]
    args.src = [os.path.abspath(s) for s in args.src]

    if args.debug:
        args.log_level = "DEBUG"

    config = Configuration(none_missing=True)
    config.configure(params=vars(args))
    return config


def summarize(result, args):
    result_list = list(result)
    klasses = result_list[0].classes
    predictions = [r.preds for r in result_list]

    # counters
    kounters = Counter()
    for p in predictions:
        kounters.update({klasses[i]: int(p > PRED_MIN) for i, p in enumerate(p)})
    for k, v in sorted(kounters.items(), key=lambda item: item[1], reverse=True):
        pc = int(v / len(predictions) * 100)
        if pc < 10:  # min 10%
            continue
        log.info(f"{k}: {pc}%")

    # squeeze
    squeezed = sqeeze_outs(predictions, args)
    rz = ClassifierResultItem("squeezed summarize", None, squeezed, klasses)
    log.info(f"{rz.confident_predictions()}")
    return rz


def load_model(config):
    model = torch.load(config.model_path, weights_only=False, map_location=config.device)
    classes = getattr(model, "_classes", [])
    classes_names = {i: n for i, n in enumerate(classes)}
    model_args = getattr(model, "_args", dict())
    if isinstance(model_args, argparse.Namespace):
        model_args = vars(model_args)
    log.info(f"model: {config.model_path}")
    log.info(f"model.classes: {classes}")
    log.info(f"model.args: {model_args}")

    for p in [
        "size_crop",
        "size_resize",
        "data_normalize",
        "data_equalize",
        "data_autocontrast",
        "data_blur",
    ]:
        if not config.get(p) and p in model_args:
            config[p] = model_args[p]
            log.info(f"{p} → {model_args[p]}")

    return config, model, classes_names


def save(result, args):
    data = [
        {
            "path": r.label,
            "preds": r.pred_ids,
            "preds_str": r.pred_names,
        }
        for r in result
    ]
    with open(args.save, "w") as outfile:
        json.dump(data, outfile, indent=2, ensure_ascii=False)


def show(result, config, pagesize=100, rows=None, cols=None):
    if rows is not None and cols is not None:
        pass
    elif config.data_crop in ("s", "super") and config.show_parts:
        cols = 6
        rows = min(3, pagesize)
    elif config.data_crop in ("sm", "supermicro") and config.show_parts:
        cols = 9
        rows = min(4, pagesize)
    elif config.data_crop == "5" and config.show_parts:
        cols = 6
        rows = min(3, pagesize)
    elif config.data_crop == "10" and config.show_parts:
        cols = 11
        rows = min(6, pagesize)
    elif pagesize < 6:
        rows, cols = 1, pagesize
    elif pagesize <= 50:
        cols = min(9, int(math.sqrt(pagesize) * 1.25 + 1))
        rows = pagesize // cols + 1
    else:
        rows, cols = 7, 9

    pagesize = rows * cols
    for page in batched(result, pagesize, strict=False):
        if len(page) <= 6:
            cols, rows = len(page), 1
        elif len(page) < pagesize and not config.show_parts:
            cols = min(9, int(math.sqrt(len(page)) * 1.25 + 1))
            rows = len(page) // cols + 1
        show_page(cols, rows, page, config)


def show_page(cols, rows, results, args):
    """
    Show images with predictions using tkinter.
    """
    import tkinter as tk
    from tkinter import Frame, ttk

    import numpy as np
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.attributes("-zoomed", True)
    root.update()
    sw = root.winfo_screenwidth() - 20
    sh = root.winfo_screenheight() - 40

    pady = int(min(40, sh // 30))
    paddy = pady // 2
    cw = int(sw / cols)
    ch = int(sh / rows)
    thw = cw - paddy
    thh = int((ch + thw / 16 * 9) // 2 - paddy)

    iframe = Frame(root)
    iframe.pack(fill="both", expand=True)

    for col in range(cols):
        iframe.grid_columnconfigure(col, weight=1)
    for row in range(rows):
        iframe.grid_rowconfigure(row, weight=1)

    log.debug(f"showing {len(results)} images, {cols=}x{rows=}, {thw=}x{thh=}, {sw=}x{sh=}")

    for i, rz in enumerate(results):
        row_idx, col_idx = divmod(i, cols)
        if rz.is_part is None:
            title = f"№{rz.no}: {rz.confident_predictions()}"
        else:
            title = f"№{rz.no}+{rz.is_part}: {rz.confident_predictions()}"

        img = None
        if args.show_original and rz.is_part is None:
            img = orig_keeper.pop(rz.label)
        else:
            if isinstance(rz.data, torch.Tensor):
                img_np = rz.data[0].cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np.clip(0, 1) * 255).astype(np.uint8)
                img = Image.fromarray(img_np)
            elif isinstance(rz.data, Image.Image):
                img = rz.data

        if not img:
            continue

        cell = Frame(iframe, width=cw, height=ch)
        cell.grid(row=row_idx, column=col_idx, padx=0, pady=0)

        label_widget = ttk.Label(cell, text=title, wraplength=thw, font=("Arial", 10))
        label_widget.pack(side="top", expand=True)

        iw, ih = img.size
        ir = iw / ih
        if args.show_original_squeeze:
            ir = max(0.92, min(1.78, ir))
        img = img.resize((int(thh * ir), thh), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        img_widget = ttk.Label(cell, image=photo)
        img_widget._image = photo
        img_widget.pack(side="top", expand=True)

    def on_key(event):
        if event.keysym == "Escape":
            sys.exit(0)
        elif event.char and event.char.isascii():
            root.destroy()

    root.bind("<Key>", on_key)
    root.mainloop()


if __name__ == "__main__":
    main()
