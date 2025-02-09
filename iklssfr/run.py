import argparse
import os
import random
from argparse import Namespace
import math

import torch
from PIL import Image
from torchvision.transforms import v2
from helpers import str_eclipse

SIZE_CROP = 600
SIZE_RESIZE = 600
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

PRED_MIN = 0.1
PRED_DEV = 2.7


def classify(model, dataloader, classes, args):
    class_names = {i: n for i, n in enumerate(classes)}
    with torch.no_grad():
        model.eval()
        for i, (input, label) in enumerate(dataloader):
            lbl = str_eclipse(os.path.basename(label), 40)
            if not isinstance(input, (tuple, list)):
                input = (input,)
            outs = None
            for j, img in enumerate(input):
                out = model(img)
                if outs is None:
                    outs = out
                else:
                    outs += out
                if len(input) > 1:
                    pred_ids, pred_names = preds_by_out(out[0], class_names)
                    print(f"#{i}+{j}: [{lbl}]: {pred_names}")
            outs /= len(input)
            pred_ids, pred_names = preds_by_out(outs[0], class_names)
            result = (input[-1][0], label, pred_ids, pred_names)
            print(f"#{i}: [{lbl}]: {pred_names}")
            yield result


def preds_by_out(out, names):
    outs = sorted(enumerate(out), key=lambda x: -x[1])
    pred_ids = {x[0]: round(float(x[1]), 3) for x in outs}
    pred_names = {names.get(p, p): w for p, w in pred_ids.items()}
    return pred_ids, pred_names


def preds_to_str(preds, pmin=PRED_MIN, pdev=PRED_DEV, limit=100):
    preds_pmin = list(filter(lambda x: x[1] > pmin, preds.items()))
    if not preds_pmin:
        return ""
    selected = [
        preds_pmin[0],
    ]
    for n, w in preds_pmin[1:]:
        if w > pmin and w > selected[0][1] / pdev:
            selected.append((n, w))
    return ",".join([n for (n, w) in selected[:limit]])


def main():
    args = read_args()

    model = torch.load(args.model_path, weights_only=False)
    classes = getattr(model, "_classes", [])
    model_args = vars(getattr(model, "_args", Namespace()))
    print(f"model: {args.model_path}")
    print(f"model.classes: {classes}")
    print(f"model.args: {model_args}")

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
            print(f"{p}→ {model_args[p]}")

    images = get_data(args)
    print(f"data: {len(images)} images")
    dataloader = get_dataloader(images, args)

    result = classify(model, dataloader, classes, args)

    if args.show:
        show(result, args=args, datasize=len(images))

    if args.save:
        save(result, args)

    return result


def save(result, args):
    raise Exception("Not implemented yet")


def show(result, args, datasize=100, N=None, M=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["toolbar"] = "None"

    if N is not None and M is not None:
        pass
    elif datasize < 6:
        N, M = 1, datasize
    elif datasize < 50:
        M = min(9, int(math.sqrt(datasize) * 1.6 + 1))
        N = datasize // M + 1
    else:
        N, M = 5, 10

    images = 0
    for i, r in enumerate(result):
        images += 1
        if images > N * M:
            plt.tight_layout()
            plt.show()
            plt.close()
            images = 1
        ax = plt.subplot(N, M, images)
        ax.axis("off")
        ax.set_title(f"{i}: {preds_to_str(r[3])}")
        if args.show_original:
            img = load_image(r[1])
            if args.show_original_squeeze:
                iw, ih = img.size
                r = max(0.66, min(1.1, ih / iw))
                img = img.resize((args.size_crop, int(r * args.size_crop)))
        else:
            img = r[0].numpy().transpose(1, 2, 0)
            img = img.clip(0, 1)
        ax.imshow(img)
    if images:
        plt.tight_layout()
        plt.show()


def load_image(image_path, transforms=None):
    try:
        img = Image.open(image_path)
        if transforms:
            img = transforms(img)
        if isinstance(img, torch.Tensor):
            img.unsqueeze_(0)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        img = None
    return img


def get_data(args):
    img_paths = []
    if os.path.isdir(args.data):
        for root, dirs, files in os.walk(args.data):
            for file in files:
                img_path = os.path.join(root, file)
                img_paths.append(img_path)

    elif os.path.isfile(args.data):
        img_paths.append(args.data)

    if args.limit:
        random.shuffle(img_paths)
        img_paths = img_paths[: args.limit]

    img_paths.sort()
    return img_paths


def get_dataloader(img_paths, args):

    transforms = get_transforms(args)
    for ipath in img_paths:
        img = load_image(ipath, transforms)
        if img is not None:
            #  data.append((img, ipath))  # 2DO: yield
            yield img, ipath


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
    if args.data_crop == "5":
        transforms.append(v2.Resize(size=args.size_resize))
        transforms.append(v2.FiveCrop(size=args.size_crop))
        transforms.append(v2.Lambda(lambda c: c.unsqueeze(0)))
    elif args.data_crop in ("c", "center"):
        transforms.append(v2.Resize(size=args.size_resize))
        transforms.append(v2.CenterCrop(size=args.size_crop))
    else:
        transforms.append(v2.Resize(size=(args.size_resize, args.size_resize)))
        transforms.append(v2.CenterCrop(size=args.size_crop))

    return v2.Compose(transforms)


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
        choices=["", "c", "center", "5"],
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

    args = parser.parse_args()

    if args.show_original:
        args.show = True

    if args.size_resize and (args.size_crop > args.size_resize or args.size_crop == 0):
        args.size_crop = args.size_resize

    for k, v in vars(args).items():
        print(f"{k}={v}")

    return args


if __name__ == "__main__":
    main()
