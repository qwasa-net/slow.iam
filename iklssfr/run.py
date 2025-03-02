import argparse
import json
import logging
import math
import os
import random
import subprocess
import tempfile
from argparse import Namespace
from collections import Counter

import torch
from helpers import AccumulatorIterator, Keeper, SuperCrop, SuperMiniCrop, setup_logging, str_eclipse
from PIL import Image
from torchvision.transforms import v2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

PRED_MIN = 0.01
PRED_DEV = 3.45

VIDEO_STEP_SEC = 59
VIDEO_CAPS_LIMIT = 99

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mkv", ".webm", ".mov", ".flv", ".wmv", ".ts"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

MPV_EXE_PATH = "/usr/bin/mpv"
FFMPEG_EXE_PATH = "/usr/bin/ffmpeg"

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
    for i, (img_group, label) in enumerate(dataloader):

        # tupalize single tensor
        if not isinstance(img_group, (tuple, list)):
            img_group = (img_group,)

        # keep here every part prediction
        outs = []

        for j, img in enumerate(img_group):
            out = model(img)[0]
            outs.append(list(out))
            if len(img_group) > 1:
                rz = ClassifierResultItem(label, img, out, classes, part=j)
                log.debug(f"#{i}+{j}: {rz}")
                if args.show_parts:
                    yield rz

        # combine all part predictions into one
        rz = ClassifierResultItem(label, img, sqeeze_outs(outs, args), classes)
        log.info(f"#{i}: {rz}")

        # send current result to the caller
        yield rz


class ClassifierResultItem:

    def __init__(self, label, data, preds, classes, part=None):
        assert len(preds) == len(classes), "preds and classes must have the same length"
        assert isinstance(label, str), "label must be a string"
        assert isinstance(preds, (list, torch.Tensor)), "preds must be a list"
        assert isinstance(classes, dict), "classes must be a dict"
        self.data = data
        self.label = label
        self.preds = list(preds) if isinstance(preds, torch.Tensor) else preds
        self.classes = classes
        self.is_part = part

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
    combined_mean = [sum(cs) / len(combined) for cs in zip(*combined)]

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
    max_dmpd = [max(cs) / len(combined) for cs in zip(*combined)]

    # combine average detected confidence and the damped max (magic!)
    combined = [max(combined_mean[j], det_mean[j]) + max_dmpd[j] for j in range(len(combined[0]))]
    # log.debug(f"{combined_mean=} / {det_mean=} / {max_dmpd=} / {combined=}")

    return combined


def get_data_paths(args):
    paths = []
    has_video = False
    if os.path.isdir(args.data):
        for root, _dirs, filenames in os.walk(args.data):
            for filename in filenames:
                _, fext = os.path.splitext(filename)
                if fext.lower() not in VIDEO_EXTENSIONS + IMAGE_EXTENSIONS:
                    continue
                if fext.lower() in VIDEO_EXTENSIONS:
                    has_video = True
                path = os.path.join(root, filename)
                paths.append(path)
    elif os.path.isfile(args.data):
        _, fext = os.path.splitext(args.data)
        if fext.lower() in VIDEO_EXTENSIONS:
            has_video = True
        paths.append(args.data)
    if args.limit:
        random.shuffle(paths)
        paths = paths[: args.limit]
    paths.sort()
    return paths, has_video


def get_dataloader(img_paths, args):
    transforms = get_transforms(args)
    for ipath in img_paths:
        fname, fext = os.path.splitext(os.path.basename(ipath))
        if fext.lower() in VIDEO_EXTENSIONS:
            yield from get_video_data_ffmpeg(ipath, transforms, args)
            # yield from get_video_data(ipath, transforms)
        elif fext.lower() in IMAGE_EXTENSIONS:
            yield from get_image_data(ipath, transforms)
        else:
            log.error(f"Unsupported file type: {fname}.{fext}")


def get_image_data(img_path, transforms):
    img = load_image(img_path, transforms)
    if img is not None:
        yield img, img_path


def get_video_data(vpath, transforms):
    """
    why is this so slow?
    """
    import cv2  # noqa

    try:
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 100)
        fstep = int(fps * VIDEO_STEP_SEC)
        fcount = 0
        capped = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            fcount += 1
            if fcount % fstep != 1:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            if transforms:
                img = transforms(img)
            if isinstance(img, torch.Tensor):
                img.unsqueeze_(0)
            yield img, vpath
            capped += 1
            if capped >= VIDEO_CAPS_LIMIT:
                break
    except Exception as e:
        log.error(f"video reading error {str_eclipse(vpath)}: {e}")
    finally:
        cap.release()


def get_video_data_ffmpeg(vpath, transforms, args):

    def ffmpeg_cmd(vpath, tmpdir):
        tmpfile_out = os.path.join(tmpdir, "f%06d.jpg")
        return [
            FFMPEG_EXE_PATH,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            vpath,
            "-vf",
            f"fps=1/{args.data_video_step}",
            tmpfile_out,
        ]

    def mpv_cmd(vpath, tmpdir, sstep=None):
        return [
            MPV_EXE_PATH,
            "--really-quiet",
            "--untimed",
            "--no-correct-pts",
            "--hr-seek=no",
            "--hr-seek-framedrop=yes",
            "--no-audio",
            "--slang=",
            "--vo=image",
            "--vo-image-format=jpg",
            "--vo-image-jpeg-quality=98",
            f"--vo-image-outdir={tmpdir}",
            f"--sstep={sstep or args.data_video_step}",
            f"--frames={args.data_video_limit}",
            vpath,
        ]

    # make temp dir
    with tempfile.TemporaryDirectory(suffix="-iklssfr", ignore_cleanup_errors=True) as tmpdir:
        os.chmod(tmpdir, 0o700)

        file_size = os.path.getsize(vpath)
        if file_size < 0.25 * 1024 * 1024:
            return  # skip small files
        if file_size < 500 * 1024 * 1024:
            sstep = args.data_video_step // 3
        else:
            sstep = args.data_video_step

        cmd = mpv_cmd(vpath, tmpdir, sstep)  # ffmpeg_cmd(vpath, tmpdir)

        try:
            log.info("calling caps maker: %s", cmd)
            subprocess.run(cmd, check=True)
        except Exception as e:
            log.error(f"ffmpeg error {str_eclipse(vpath)}: {e}")
            return

        paths = []
        for root, _dirs, filenames in os.walk(tmpdir):
            for filename in filenames:
                path = os.path.join(root, filename)
                paths.append(path)
        paths.sort()

        prev_file_size = None
        for path in paths:
            file_size = os.path.getsize(path)
            if file_size == prev_file_size:
                continue  # skip duplicates, detected by size (¿bug?)
            prev_file_size = file_size
            yield from get_image_data(path, transforms)


def load_image(image_path, transforms=None):
    try:
        img = Image.open(image_path)
        orig_keeper.put(image_path, img)
        if transforms:
            img = transforms(img)
        if isinstance(img, torch.Tensor):
            img.unsqueeze_(0)
    except Exception as e:
        log.error(f"Error loading image {image_path}: {e}")
        img = None
    return img


def get_transforms(args):
    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    if args.data_normalize:
        transforms.append(v2.Normalize(IMG_MEAN, IMG_STD))
    if args.data_autocontrast:
        transforms.append(v2.functional.autocontrast)
    if args.data_equalize:
        transforms.append(v2.functional.equalize)
    if args.data_blur:
        transforms.append(v2.GaussianBlur(kernel_size=(3, 7), sigma=0.5))
    if args.data_crop in ("5", "5+"):
        transforms.append(v2.Resize(size=int(args.size_resize * 1.33)))
        transforms.append(v2.FiveCrop(size=args.size_crop))
        transforms.append(v2.Lambda(lambda c: c.unsqueeze(0)))
    elif args.data_crop in ("10", "10+"):
        transforms.append(v2.Resize(size=int(args.size_resize * 1.33)))
        transforms.append(v2.TenCrop(size=args.size_crop))
        transforms.append(v2.Lambda(lambda c: c.unsqueeze(0)))
    elif args.data_crop in ("s", "super"):
        transforms.append(SuperCrop(args.size_crop, args.size_resize))
        transforms.append(v2.Lambda(lambda c: c.unsqueeze(0)))
    elif args.data_crop in ("sm", "supermini"):
        transforms.append(SuperMiniCrop(args.size_crop, args.size_resize))
        transforms.append(v2.Lambda(lambda c: c.unsqueeze(0)))
    elif args.data_crop in ("c", "center"):
        transforms.append(v2.Resize(size=args.size_resize))
        transforms.append(v2.CenterCrop(size=args.size_crop))
    else:
        transforms.append(v2.Resize(size=(args.size_resize, args.size_resize)))
        transforms.append(v2.CenterCrop(size=args.size_crop))
    return v2.Compose(transforms)


def main():

    args = read_args()
    setup_logging(args)

    log.info("args:")
    for k, v in vars(args).items():
        log.info(f"{k}={v}")

    args, model, classes = load_model(args)

    data_paths, has_video = get_data_paths(args)
    log.info(f"found data: {len(data_paths)=}, {has_video=}")
    dataloader = get_dataloader(data_paths, args)

    orig_keeper.keep = args.show_original

    class_results = classify(model, dataloader, classes, args)
    result = AccumulatorIterator(class_results)

    if args.show:
        show(
            result,
            args=args,
            pagesize=len(data_paths) if not has_video else 25 * len(data_paths),
        )

    if args.summarize:
        summarize(result, args)

    if args.save:
        save(result, args)


def read_args():
    parser = argparse.ArgumentParser()
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
        "--data",
        "-d",
        type=str,
        default="data/",
    )
    parser.add_argument(
        "--data-video-step",
        type=int,
        default=VIDEO_STEP_SEC,
    )
    parser.add_argument(
        "--data-video-limit",
        type=int,
        default=VIDEO_CAPS_LIMIT,
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
        "--log",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
    )

    args = parser.parse_args()

    if args.show_original:
        args.show = True

    if args.size_resize and (args.size_crop > args.size_resize or args.size_crop == 0):
        args.size_crop = args.size_resize

    if args.data and args.data.startswith("file://"):
        args.data = args.data[7:]

    return args


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


def load_model(args):
    model = torch.load(args.model_path, weights_only=False)
    classes = getattr(model, "_classes", [])
    classes_names = {i: n for i, n in enumerate(classes)}
    model_args = vars(getattr(model, "_args", Namespace()))
    log.info(f"model: {args.model_path}")
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
        if not getattr(args, p) and p in model_args:
            setattr(args, p, model_args[p])
            log.info(f"{p} → {model_args[p]}")

    return args, model, classes_names


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


def show(result, args, pagesize=100, rows=None, cols=None):
    """
    Show images with predictions.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    cnt = 1
    mpl.rcParams["toolbar"] = "None"
    if rows is not None and cols is not None:
        pass
    elif args.data_crop in ("s", "super") and args.show_parts:
        cols = 6
        rows = min(4, pagesize)
        cnt = 6
    elif args.data_crop in ("sm", "supermicro") and args.show_parts:
        cols = 8
        rows = min(4, pagesize)
        cnt = 4
    elif args.data_crop == "5" and args.show_parts:
        cols = 6
        rows = min(4, pagesize)
        cnt = 6
    elif args.data_crop == "10" and args.show_parts:
        cols = 11
        rows = min(5, pagesize)
        cnt = 11
    elif pagesize < 6:
        rows, cols = 1, pagesize
    elif pagesize <= 50:
        cols = min(9, int(math.sqrt(pagesize) * 1.25 + 1))
        rows = pagesize // cols + 1
    else:
        rows, cols = 6, 10

    images_on_page = 0
    for i, rz in enumerate(result):
        images_on_page += 1
        if images_on_page > rows * cols:
            plt.tight_layout()
            plt.show()
            plt.close()
            images_on_page = 1
        ax = plt.subplot(rows, cols, images_on_page)
        ax.axis("off")
        ax.set_title(f"№{i//cnt}: {rz.confident_predictions()}")
        if args.show_original and rz.is_part is None:
            img = orig_keeper.pop(rz.label)
            if args.show_original_squeeze:
                iw, ih = img.size
                rz = max(0.66, min(1.1, ih / iw))
                img = img.resize((args.size_crop, int(rz * args.size_crop)))
        else:
            if isinstance(rz.data, torch.Tensor):
                img = rz.data[0].cpu().numpy().transpose(1, 2, 0)
                img = img.clip(0, 1)
            else:
                img = rz.data
        ax.imshow(img)
    if images_on_page:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
