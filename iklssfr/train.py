import argparse
import configparser
import logging
import os
import random
import time
import urllib.parse
from datetime import datetime as dt

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import functional as trf
from torchvision.transforms import v2 as trv2

import ihelp.helpers as helpers
from ihelp.config import Configuration
from ihelp.helpers import log_tz, setup_logging, sizexx
from ihelp.media import IMAGE_EXTENSIONS

from .datasets import MultiLabelDataset
from .transforms import RandomResizedCropSq, RandomRotateChoice

EPOCHS_LIMIT = 19
DATA_CROP = 224
DATA_RESIZE = 242
BATCH_SIZE = 12
DATA_TRAIN_PC = 91
DATA_RELOAD = 7
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GITHUB_REPO = "pytorch/vision"
ACC_WEIGHTS = [0.33, 0.66]

log = logging.getLogger()


class Trainer:
    SIGMOID_THRESHOLD = 0.55
    OPTIMIZERS = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    CRITERIONS = {"bce": torch.nn.BCEWithLogitsLoss, "ce": torch.nn.CrossEntropyLoss}

    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.setup_model()

    def __str__(self):
        return f"Model({self.model}, {self.model_weights}, {self.model_source}, {self.model_keep_classes})"

    def setup_model(self, criterion=None, optimizer=None):
        """
        Sets up the model, criterion, optimizer, and scheduler.
        """

        self.model = self.load_model()
        if not self.config.model_keep_classes:
            self.adjust_model_classifier_layer()

        self.model.to(self.config.device)

        self.criterion = self.CRITERIONS[self.config.criterion](reduction="mean")

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = self.OPTIMIZERS[self.config.optimizer](params, lr=self.config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.5, last_epoch=-1)

    def adjust_model_classifier_layer(self):
        """
        Adjust model for the number of classes.
        """
        if not len(self.data.classes):
            self.data.load(0)
        num_classes = len(self.data.classes)
        if hasattr(self.model, "fc"):
            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, num_classes)
            log.info("fc layer > %s classes", num_classes)
            return
        if hasattr(self.model, "classifier") and isinstance(self.model.classifier, torch.nn.Sequential):
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[-1] = torch.nn.Linear(num_ftrs, num_classes)
            log.info("classifier > %s classes", num_classes)
            return

    def train(self):
        """
        Fine-tunes a model on a dataset and saves the most accurate version.
        """

        best_acc, eval_acc = 0.0, 0.0
        best_path = None

        for epoch in range(self.config.epochs_limit):
            log.info("epoch №%d/%d", epoch + 1, self.config.epochs_limit)

            # reload data (if needed)
            self.data.load(epoch)

            # train
            tm_start = time.time()
            train_acc = self.train_model_one_epoch()
            tm_secs = time.time() - tm_start
            log.info("train acc=%.2f; %d:%02d", train_acc, tm_secs // 60, tm_secs % 60)

            # eval or not eval
            if self.data.ds_eval_len > 0:
                eval_acc = self.eval_model()
                epoch_acc = sum((x * y for x, y in zip((train_acc, eval_acc), ACC_WEIGHTS, strict=False)))
            else:
                epoch_acc = train_acc

            # save the model (best so far or all)
            if epoch_acc > best_acc or self.config.save_all:
                best_path = self.save_model(epoch, epoch_acc)
            best_acc = max(best_acc, epoch_acc)

        return best_acc, best_path

    def train_model_one_epoch(self):
        """
        Trains a model for one epoch.
        """

        corrects = 0
        step_every_n = self.config.optimizer_step_every_n
        self.model.train()
        self.optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(self.data.dl_train, start=1):
            is_last = i == len(self.data.dl_train)
            if i % 33 == 1 or is_last:
                log.info(" · batch №%d, inputs: %s, labels: %s", i, sizexx(inputs), sizexx(labels))
            inputs, labels = inputs.to(self.config.device), labels.float().to(self.config.device)
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                if is_last or (step_every_n <= 1) or (i % step_every_n == 0):
                    log.debug(" ·· stepping on accumulated loss: %.3f", loss.item())
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                log_tz(" ·· ", inputs=inputs, outputs=outputs)
            outputs.sigmoid_()
            preds = outputs > self.SIGMOID_THRESHOLD  # multi-label (multi-hot)
            corrects += torch.eq(preds, labels.data).sum()
        self.scheduler.step()

        return corrects / (self.data.ds_train_len * len(self.data.classes))

    def eval_model(self):
        """
        Evaluates a model on a dataset.
        """

        losses, corrects = 0.0, 0

        with torch.set_grad_enabled(False):
            self.model.eval()
            for inputs, labels in self.data.dl_eval:
                inputs, labels = inputs.to(self.config.device), labels.float().to(self.config.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.float())
                losses += loss.item()
                outputs.sigmoid_()
                preds = outputs > self.SIGMOID_THRESHOLD
                corrects += torch.eq(preds, labels.data).sum()

        eval_loss = losses / (self.data.ds_eval_len * len(self.data.classes))
        eval_acc = int(corrects) / (self.data.ds_eval_len * len(self.data.classes))
        log.info("eval: loss=%.2f acc=%.2f", eval_loss, eval_acc)

        return eval_acc

    def load_model(self):
        assert self.config.model_source in ["file", "github"]

        if self.config.model_source == "file":
            return torch.load(self.config.model, weights_only=False)

        if self.config.model_source == "github":
            return torch.hub.load(
                GITHUB_REPO,
                source="github",
                model=self.config.model,
                weights=self.config.model_weights,
                skip_validation=True,
                trust_repo=True,
            )
        return None

    def save_model(self, epoch, acc=None):
        """
        Saves the model to a file.
        """
        model_path = self.config.model_path.format(
            epoch=(epoch + 1),
            last=(epoch == self.config.epochs_limit - 1),
            acc=acc,
        )
        self.model._classes = self.data.classes
        self.model._args = self.config.as_dict()
        self.model._epoch = epoch
        self.model._ds_train_len = self.data.ds_train_len
        torch.save(self.model, model_path)
        log.info("model is saved to `%s`", model_path)
        return model_path


class Data:
    classes = []
    dataset = None
    dataset_len = 0
    dl_train = None
    dl_eval = None

    def __init__(self, transforms, config):
        self.config = config
        self.transforms = transforms
        if self.config.data_classes:
            self.classes = self.config.data_classes.copy()

    def load(self, epoch=None, reload=False):
        """ """

        if not self.dataset or reload:
            self.load_dataset()

        if (
            not reload
            and self.dl_train is not None
            and self.config.data_reload > 0
            and epoch is not None
            and epoch % self.config.data_reload != 0
        ):
            return

        self.ds_train, self.ds_eval = self.split_dataset()
        self.ds_train_len, self.ds_eval_len = len(self.ds_train), len(self.ds_eval)

        self.dl_train = self.create_dataloader(self.ds_train, shuffle=True)
        self.dl_eval = self.create_dataloader(self.ds_eval, shuffle=False)

        log.info("train sub-dataset size: %d", self.ds_train_len)
        log.info("eval sub-dataset size: %d", self.ds_eval_len)
        log.info("train sub-dataset classes=%s", self.classes)

    def load_dataset(self):
        """
        Loads a dataset from a directory.
        """

        self.dataset = torchvision.datasets.DatasetFolder(
            root=self.config.src,
            allow_empty=True,
            loader=helpers.tenzor_loader,
            extensions=IMAGE_EXTENSIONS,
        )
        if not self.classes:
            self.classes = self.dataset.classes.copy()
        self.dataset_len = len(self.dataset)
        log.info("loaded dataset: %s, %d items", self.config.src, self.dataset_len)

    def split_dataset(self):
        """
        Splits dataset it into train and eval subsets.
        """

        idxs = list(range(self.dataset_len))
        random.shuffle(idxs)
        idx_split = int(self.dataset_len * self.config.data_train_pc / 100)

        dataset_train = Subset(
            MultiLabelDataset(
                dataset=self.dataset,
                transform=trv2.Compose(self.transforms.train) if self.transforms.train else None,
                classes=self.classes,
                drop_prefix=self.config.src,
            ),
            indices=idxs[:idx_split],
        )

        dataset_eval = Subset(
            MultiLabelDataset(
                dataset=self.dataset,
                transform=trv2.Compose(self.transforms.eval) if self.transforms.eval else None,
                classes=self.classes,
                drop_prefix=self.config.data,
            ),
            indices=idxs[idx_split:],
        )

        return dataset_train, dataset_eval

    def create_dataloader(self, dataset, shuffle=True):
        """
        Creates dataloaders for the dataset.
        """

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.workers,
            persistent_workers=True,
        )


class Transforms:
    train = []
    eval = []

    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    def __init__(self, args):
        """
        Initializes transforms for the dataset.
        """

        self.args = args
        if not self.args.data_ready:
            self.train = self.trans_train()
            self.eval = self.trans_eval()
        else:
            self.train = None
            self.eval = None

    def trans_train(self):
        trans = []

        if self.args.data_flip:
            trans += [
                trv2.RandomHorizontalFlip(p=0.2),
                trv2.RandomVerticalFlip(p=0.2),
                RandomRotateChoice(angles=[0, 0, 0, 0, 11, 23, -23, 180]),
            ]

        if self.args.data_crop == "random-resized":  # crop and resize
            trans.append(
                RandomResizedCropSq(
                    size=self.args.size_crop,
                    scale=(0.8, 1.0),
                    ratio=(0.5, 2.2),
                )
            )
        else:  # resize and crop
            trans.append(
                trv2.RandomChoice(
                    [
                        # ignore aspect ratio -- distorted square image
                        trv2.Resize(size=(self.args.size_resize, self.args.size_resize)),
                        # keep aspect ratio
                        trv2.Resize(size=self.args.size_resize),
                    ]
                ),
            )
            if self.args.data_crop == "random":
                trans.append(trv2.RandomCrop(size=self.args.size_crop))
            else:
                trans.append(trv2.CenterCrop(size=self.args.size_crop))

        if self.args.data_autocontrast:
            trans.append(trf.autocontrast)

        if self.args.data_normalize:
            trans.append(trv2.Normalize(self.IMG_MEAN, self.IMG_STD))

        if self.args.data_equalize:
            trans.append(trf.equalize)

        return trans

    def trans_eval(self):
        trans = [
            trv2.RandomChoice(
                [
                    # ignore aspect ratio -- distorted square image
                    trv2.Resize(size=(self.args.size_resize, self.args.size_resize)),
                    # keep aspect ratio
                    trv2.Resize(size=self.args.size_resize),
                ],
                p=[1, 1],
            ),
            trv2.CenterCrop(size=self.args.size_crop),
        ]
        if self.args.data_normalize:
            trans.append(trv2.Normalize(self.IMG_MEAN, self.IMG_STD))
        if self.args.data_autocontrast:
            trans.append(trf.autocontrast)
        if self.args.data_equalize:
            trans.append(trf.equalize)
        return trans


def main():
    """
    Somewhere we have to start. Let's start here.
    """

    # read cli options
    config = read_args()
    setup_logging(config)

    log.info("the train is departing right now...")
    for k, v in config.items():
        log.info("%s=%s", k, v)

    # train
    tm_start = time.time()
    model = Trainer(config, Data(Transforms(config), config))
    accuracy, path = model.train()
    tm_secs = int(time.time() - tm_start)

    # log results
    log.info("the train has arrived at (%s)", path)
    log.info("total travel time: %d:%02d", tm_secs // 60, tm_secs % 60)
    log.info("best achieved accuracy: %.4f", accuracy)


def read_args():
    """
    Parses command-line arguments.
    """

    ts = dt.now().strftime("%Y%m%d%H%M%S")

    parser = argparse.ArgumentParser(description="Train a model on a images dataset (multiple-labels classification)")

    сparser = argparse.ArgumentParser(add_help=False)
    сparser.add_argument("--config", default="settings.ini")
    сparser.add_argument("--config-section", default="DEFAULT")
    args, remaining_argv = сparser.parse_known_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    def cfg(key, default=None):
        return config.get(args.config_section, key, fallback=default)

    def cfg_bool(key, default=None) -> bool:
        return str(cfg(key, default)).lower() in ["true", "yes", "on"]

    parser.add_argument(
        "--name",
        type=str,
        default=cfg("name", "iklssfr"),
    )
    parser.add_argument(
        "src",
        type=str,
        nargs="?",
        default=cfg("src", "./data/"),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=cfg("model-path"),
    )
    parser.add_argument(
        "--model-path-dir",
        type=str,
        default=cfg("model-path-dir", ""),
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        default=cfg_bool("save-all", "False"),
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=cfg("model", "resnet18"),
        help="resnet18|resnet50|…",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default=cfg("model-weights", "DEFAULT"),
        help="DEFAULT|IMAGENET1K_V2|…",
    )
    parser.add_argument(
        "--model-source",
        type=str,
        default=cfg("model-source", "github"),
        choices=["github", "file"],
    )
    parser.add_argument(
        "--model-keep-classes",
        action="store_true",
        default=cfg_bool("model-keep-classes", "False"),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(cfg("learning-rate", "0.001") or 0.001),
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=cfg("optimizer", "adamw"),
        choices=["adam", "adamw", "sgd"],
    )
    parser.add_argument(
        "--optimizer-step-every-n",
        type=int,
        default=int(cfg("optimizer-step-every-n", "3") or 3),
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default=cfg("criterion", "bce"),
        choices=["bce"],
    )
    parser.add_argument(
        "--data-classes",
        type=str,
        default=cfg("data-classes", ""),
    )
    parser.add_argument(
        "--data-ready",
        action="store_true",
        default=cfg_bool("data-ready", "False"),
    )
    parser.add_argument(
        "--data-normalize",
        "-dn",
        action="store_true",
        default=cfg_bool("data-normalize", "True"),
    )
    parser.add_argument(
        "--data-equalize",
        "-dq",
        action=argparse.BooleanOptionalAction,
        default=cfg_bool("data-equalize", "False"),
    )
    parser.add_argument(
        "--data-autocontrast",
        "-da",
        action=argparse.BooleanOptionalAction,
        default=cfg_bool("data-autocontrast", "False"),
    )
    parser.add_argument(
        "--data-flip",
        "-df",
        action=argparse.BooleanOptionalAction,
        default=cfg_bool("data-flip", "True"),
    )
    parser.add_argument(
        "--data-crop",
        "-dc",
        type=str,
        default=cfg("data-crop", "center"),
        choices=["random", "random-resized", "center"],
    )
    parser.add_argument(
        "--data-reload",
        type=int,
        default=int(cfg("data-reload", DATA_RELOAD)),
    )
    parser.add_argument(
        "--data-train-pc",
        type=int,
        default=int(cfg("data-train-pc", DATA_TRAIN_PC) or DATA_TRAIN_PC),
    )
    parser.add_argument(
        "--size-crop",
        type=int,
        default=int(cfg("size-crop", DATA_CROP) or DATA_CROP),
    )
    parser.add_argument(
        "--size-resize",
        type=int,
        default=int(cfg("size-resize", DATA_RESIZE) or DATA_RESIZE),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(cfg("batch-size", BATCH_SIZE) or BATCH_SIZE),
    )
    parser.add_argument(
        "--epochs-limit",
        type=int,
        default=int(cfg("epochs-limit", EPOCHS_LIMIT) or EPOCHS_LIMIT),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=cfg("device", str(DEVICE)) or DEVICE,
    )
    parser.add_argument(
        "--log",
        type=str,
        default=cfg("log"),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=cfg("log-level", "INFO"),
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=cfg_bool("debug", "False"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(cfg("workers", "0")) or ((os.cpu_count() or 4) // 4),
    )

    args = parser.parse_args(remaining_argv)

    if args.src.startswith("file://"):
        args.src = urllib.parse.unquote(args.src[7:])

    if args.size_resize and args.size_crop > args.size_resize:
        args.size_crop = args.size_resize

    if args.data_train_pc < 0 or args.data_train_pc > 100:
        args.data_eval_pc = DATA_TRAIN_PC

    if args.data_classes:
        args.data_classes = args.data_classes.split(",")

    if args.data_ready:
        args.data_normalize = True
        args.data_autocontrast = True

    if args.model_path:
        args.model_path = str(args.model_path).format(name=args.name, ts=ts, model=args.model)
    else:
        if args.save_all:
            args.model_path = f"{args.name}_{ts}_{{epoch}}.pth"
        else:
            args.model_path = f"{args.name}_{ts}.pth"

    if args.log is None:
        args.log = f"{args.name}_{ts}.log"

    if args.debug:
        args.log_level = "DEBUG"

    if args.model_path_dir:
        args.model_path = os.path.join(args.model_path_dir, args.model_path)
        args.log = os.path.join(args.model_path_dir, args.log)

    config = Configuration(none_missing=True)
    config.configure(params=vars(args))

    return config


if __name__ == "__main__":
    main()
