import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from helpers import DatasetTransform
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2

EPOCHS_LIMIT = 19
DATA_CROP = 224
DATA_RESIZE = 242
BATCH_SIZE = 12
DATA_TRAIN_PC = 81
DATA_RELOAD = 6
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
GITHUB_REPO = "pytorch/vision"
ACC_WEIGHTS = [0.2, 0.8]


def main():
    """
    Somewhere we have to start. Let's start here.
    """

    # read cli options
    args = read_args()
    setup_logging(args)

    logging.info("the train is departing right now...")
    for k, v in vars(args).items():
        logging.info(f"{k}={v}")

    # train
    tm_start = time.time()
    accuracy, path, classes = train(args)
    tm_secs = int(time.time() - tm_start)

    # log results
    logging.info(f"the train has arrived at ({path})")
    logging.info(f"total travel time: {tm_secs // 60:.0f}:{tm_secs % 60:.0f}")
    logging.info(f"best achieved accuracy: {accuracy:.4f}")


def train(args):
    """
    Loads a pre-trained model and calls fine-tuning.
    """

    # load model
    if args.model_source == "file":
        model = torch.load(args.model, weights_only=False)
    elif args.model_source == "github":
        model = torch.hub.load(
            GITHUB_REPO,
            source="github",
            model=args.model,
            weights=args.model_weights,
            skip_validation=True,
            trust_repo=True,
        )
    else:
        raise ValueError(f"unknown model source: {args.model_source}")

    rc = train_model(
        model,
        args,
    )

    return rc


def train_model(
    model,
    args,
):
    """
    Fine-tunes a model on a dataset and saves the most accurate version.
    """

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(params, lr=0.001, momentum=0.9)
    optimizer = optim.Adam(params, lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0
    best_path = None

    for epoch in range(args.epochs_limit):

        logging.info(f"epoch #{epoch+1}/{args.epochs_limit}")

        if epoch % args.data_reload == 0:
            # reload data every N epochs
            dataloaders, classes = get_dataloaders(args)

        if epoch == 0 and not args.model_keep_classes:
            # adjust model for the number of classes
            if hasattr(model, "fc"):
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, len(classes))
            elif hasattr(model, "classifier"):
                if isinstance(model.classifier, nn.Sequential):
                    num_ftrs = model.classifier[1].in_features
                    model.classifier[-1] = nn.Linear(num_ftrs, len(classes))

        # train
        model, train_acc = train_model_one_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            scheduler,
        )

        # eval
        eval_acc = eval_model(
            model,
            dataloaders["eval"],
            criterion,
        )

        # weighted average of train and eval accuracies
        epoch_acc = sum((x * y for x, y in zip((train_acc, eval_acc), ACC_WEIGHTS)))

        if epoch_acc > best_acc or args.save_all:
            best_path = save_model(model, args, epoch, classes)

        best_acc = max(best_acc, epoch_acc)

    return best_acc, best_path, classes


def save_model(model, args, epoch, classes):
    """
    Saves the model to a file.
    """
    model_path = args.model_path.format(
        epoch=epoch,
        last=(epoch == args.epochs_limit - 1),
    )
    model._classes = classes
    model._args = args
    torch.save(model, model_path)
    logging.info(f"model is saved to `{model_path}`")
    return model_path


def train_model_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
):
    """
    Trains a model for one epoch.
    """

    loss, corrects = 0.0, 0
    model.train()
    train_len = len(dataloader.dataset)
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)  # train on hottest
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        loss += loss.item() * inputs.size(0)
        corrects += torch.sum(preds == labels.data)
    scheduler.step()

    train_loss = int(loss) / train_len
    train_acc = int(corrects) / train_len
    logging.info(f"train: loss={train_loss:.2f} acc={train_acc:.2f}")

    return model, train_acc


def eval_model(
    model,
    dataloader,
    criterion,
):
    """
    Evaluates a model on a dataset.
    """

    loss, corrects = 0.0, 0
    model.eval()
    eval_len = len(dataloader.dataset)
    if eval_len == 0:
        return 0.0
    for inputs, labels in dataloader:
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            loss = criterion(outputs, labels)
        loss += loss.item() * inputs.size(0)
        corrects += torch.sum(preds == labels.data)

    eval_loss = int(loss) / eval_len
    eval_acc = int(corrects) / eval_len
    logging.info(f"eval: loss={eval_loss:.2f} acc={eval_acc:.2f}")

    return eval_acc


def get_datasets(args):
    """
    Loads a dataset from a directory and splits it into train and eval subsets.
    """

    TRANSFORMS_TRAIN = [
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.25),
        v2.RandomVerticalFlip(p=0.25),
        v2.RandomChoice(
            [
                # ignore aspect ratio -- distorted square image
                v2.Resize(size=(args.size_resize, args.size_resize)),
                # keep aspect ratio
                v2.Resize(size=args.size_resize),
            ],
            p=[1 if args.data_crop != "center" else 0, 2],
        ),
        v2.RandomChoice(
            [
                # square crop
                v2.RandomCrop(size=args.size_crop),
                v2.CenterCrop(size=args.size_crop),
            ],
            p=[2 if args.data_crop != "center" else 0, 1],
        ),
        v2.ToDtype(torch.float32, scale=True),
    ]

    TRANSFORMS_EVAL = [
        v2.ToImage(),
        v2.Resize(
            # ignore aspect ratio
            size=(args.size_resize, args.size_resize),
        ),
        v2.CenterCrop(size=args.size_crop),
        v2.ToDtype(torch.float32, scale=True),
    ]

    if args.data_normalize:
        TRANSFORMS_TRAIN.append(v2.Normalize(IMG_MEAN, IMG_STD))
        TRANSFORMS_EVAL.append(v2.Normalize(IMG_MEAN, IMG_STD))

    if args.data_equalize:
        TRANSFORMS_TRAIN.append(v2.functional.equalize)
        TRANSFORMS_EVAL.append(v2.functional.equalize)

    if args.data_autocontrast:
        TRANSFORMS_TRAIN.append(v2.functional.autocontrast)
        TRANSFORMS_EVAL.append(v2.functional.autocontrast)

    dataset = torchvision.datasets.ImageFolder(
        root=args.data,
        transform=v2.Compose(TRANSFORMS_TRAIN),
        allow_empty=True,
    )

    classes = dataset.classes.copy()
    dataset_size = len(dataset)

    idxs = list(range(dataset_size))
    random.shuffle(idxs)

    idx_split = int(dataset_size * args.data_train_pc / 100)

    dataset_train = Subset(
        DatasetTransform(dataset, v2.Compose(TRANSFORMS_TRAIN)),
        indices=idxs[:idx_split],
    )

    dataset_eval = Subset(
        DatasetTransform(dataset, v2.Compose(TRANSFORMS_EVAL)),
        indices=idxs[idx_split:],
    )

    return (
        {
            "train": dataset_train,
            "eval": dataset_eval,
        },
        classes,
    )


def get_dataloaders(args):
    """
    Creates dataloaders for train and eval datasets.
    """

    datasets, classes = get_datasets(args)

    dataloader_train = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    dataloader_eval = DataLoader(
        datasets["eval"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    dataloaders = {
        "train": dataloader_train,
        "eval": dataloader_eval,
    }

    logging.info(f"train dataset size: {len(dataloaders['train'].dataset)}")
    logging.info(f"eval dataset size: {len(dataloaders['eval'].dataset)}")
    logging.info(f"train dataset classes={classes}")

    return dataloaders, classes


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


def read_args():
    """
    Parses command-line arguments.
    """

    ts = dt.now().strftime("%Y%m%d%H%M%S")

    parser = argparse.ArgumentParser(
        description="Train a model on a images dataset (single-label)",
    )

    parser.add_argument("--model-path", type=str, default=f"k{ts}.pth")
    parser.add_argument("--save-all", action="store_true", default=False)

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="resnet18",
        help="resnet18|resnet50|…",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default="DEFAULT",
        help="DEFAULT|IMAGENET1K_V2|…",
    )
    parser.add_argument(
        "--model-source",
        type=str,
        default="github",
        choices=["github", "file"],
    )
    parser.add_argument(
        "--model-keep-classes",
        action="store_true",
        default=False,
    )

    parser.add_argument("--data", type=str, default="data/")
    parser.add_argument(
        "--data-normalize",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--data-equalize",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--data-autocontrast",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--data-crop",
        type=str,
        default="random",
        choices=["random", "center"],
    )
    parser.add_argument("--data-reload", type=int, default=DATA_RELOAD)
    parser.add_argument("--data-train-pc", type=int, default=DATA_TRAIN_PC)

    parser.add_argument("--size-crop", type=int, default=DATA_CROP)
    parser.add_argument("--size-resize", type=int, default=DATA_RESIZE)

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs-limit", type=int, default=EPOCHS_LIMIT)

    parser.add_argument("--device", type=str, default=DEVICE)

    parser.add_argument("--log", type=str, default=f"k{ts}.log")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--workers", type=int, default=os.cpu_count() // 4)

    args = parser.parse_args()

    if args.size_resize and args.size_crop > args.size_resize:
        args.size_crop = args.size_resize

    if args.data_train_pc < 0 or args.data_train_pc > 100:
        args.data_eval_pc = DATA_TRAIN_PC

    return args


if __name__ == "__main__":
    main()
