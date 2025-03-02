import argparse
import logging
import os
import random
import time
from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from helpers import DatasetTransform, MultiLabelDataset, RandomRotateChoice, setup_logging, sizxx
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import functional as tvf
from torchvision.transforms import v2 as tvt2

EPOCHS_LIMIT = 19
DATA_CROP = 224
DATA_RESIZE = 242
BATCH_SIZE = 12
DATA_TRAIN_PC = 88
DATA_RELOAD = 7
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GITHUB_REPO = "pytorch/vision"
ACC_WEIGHTS = [0.33, 0.66]

log = logging.getLogger()


class Trainer:

    SIGMOID_THRESHOLD = 0.5

    def __init__(self, args, data, criterion=None, optimizer=None):
        self.args = args
        self.data = data
        self.setup_model(criterion, optimizer)

    def __str__(self):
        return f"Model({self.model}, {self.model_weights}, {self.model_source}, {self.model_keep_classes})"

    def setup_model(self, criterion=None, optimizer=None):
        """
        Sets up the model, criterion, optimizer, and scheduler.
        """

        self.model = self.load_model()
        if not self.args.model_keep_classes:
            self.adjust_model_classifier_layer()

        self.criterion = criterion or nn.BCEWithLogitsLoss(reduction="mean")

        params = [p for p in self.model.parameters() if p.requires_grad]
        # self.optimizer = optimizer or optim.SGD(params, lr=0.001, momentum=0.9)
        self.optimizer = optimizer or optim.AdamW(params, lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=8, gamma=0.25)

    def adjust_model_classifier_layer(self):
        """
        Adjust model for the number of classes.
        """
        if not len(self.data.classes):
            self.data.load(0)
        num_classes = len(self.data.classes)
        if hasattr(self.model, "fc"):
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            log.info("fc layer > %s classes", num_classes)
            return
        if hasattr(self.model, "classifier") and isinstance(self.model.classifier, nn.Sequential):
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
            log.info("classifier > %s classes", num_classes)
            return

    def train(self):
        """
        Fine-tunes a model on a dataset and saves the most accurate version.
        """

        best_acc, eval_acc = 0.0, 0.0
        best_path = None

        for epoch in range(self.args.epochs_limit):
            log.info("epoch №%d/%d", epoch + 1, self.args.epochs_limit)

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
                epoch_acc = sum((x * y for x, y in zip((train_acc, eval_acc), ACC_WEIGHTS)))
            else:
                epoch_acc = train_acc

            # save the model (best so far or all)
            if epoch_acc > best_acc or self.args.save_all:
                best_path = self.save_model(epoch, epoch_acc)
            best_acc = max(best_acc, epoch_acc)

        return best_acc, best_path

    def train_model_one_epoch(self, step_every_n=None):
        """
        Trains a model for one epoch.
        """

        corrects = 0
        step_every_n = self.args.batch_size if step_every_n is None else step_every_n
        self.model.train()
        self.optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(self.data.dl_train, start=1):
            is_last = i == len(self.data.dl_train)
            if i % 25 == 1 or is_last:
                log.info(" · batch №%d, inputs: %s, labels: %s", i, sizxx(inputs), sizxx(labels))
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.float())
                loss.backward()
                if is_last or (step_every_n <= 1) or (i % step_every_n == 0):
                    log.info(" ·· stepping on accumulated loss: %.3f", loss.item())
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            outputs.sigmoid_()
            preds = outputs > self.SIGMOID_THRESHOLD  # train on multi-hots
            # preds = torch.argmax(outputs, 1)  # train on hottest
            corrects += torch.eq(preds, labels.data).sum()
        self.scheduler.step()

        train_epoch_acc = corrects / (self.data.ds_train_len * len(self.data.classes))

        return train_epoch_acc

    def eval_model(self):
        """
        Evaluates a model on a dataset.
        """

        losses, corrects = 0.0, 0

        with torch.set_grad_enabled(False):
            self.model.eval()
            for inputs, labels in self.data.dl_eval:
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
        assert self.args.model_source in ["file", "github"]

        if self.args.model_source == "file":
            return torch.load(self.args.model_path, weights_only=False)

        if self.args.model_source == "github":
            return torch.hub.load(
                GITHUB_REPO,
                source="github",
                model=self.args.model,
                weights=self.args.model_weights,
                skip_validation=True,
                trust_repo=True,
            )

    def save_model(self, epoch, acc=None):
        """
        Saves the model to a file.
        """
        model_path = self.args.model_path.format(
            epoch=epoch,
            last=(epoch == self.args.epochs_limit - 1),
            acc=acc,
        )
        self.model._classes = self.data.classes
        self.model._args = self.args
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

    def __init__(self, args):
        self.args = args
        self.tr = Transforms(args)
        if self.args.data_classes:
            self.classes = self.args.data_classes.copy()

    def load(self, epoch=None, reload=False):
        """ """

        if not self.dataset or reload:
            self.load_dataset()

        if (
            not reload
            and self.dl_train is not None
            and self.args.data_reload > 0
            and epoch is not None
            and epoch % self.args.data_reload != 0
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

        self.dataset = torchvision.datasets.ImageFolder(
            root=self.args.data,
            transform=tvt2.Compose(self.tr.train),
            allow_empty=True,
        )
        if not self.classes:
            self.classes = self.dataset.classes.copy()
        self.dataset_len = len(self.dataset)
        log.info("loaded dataset: %s, %d", self.args.data, self.dataset_len)

    def split_dataset(self):
        """
        Splits dataset it into train and eval subsets.
        """

        idxs = list(range(self.dataset_len))
        random.shuffle(idxs)
        idx_split = int(self.dataset_len * self.args.data_train_pc / 100)

        dataset_train = Subset(
            MultiLabelDataset(
                dataset=DatasetTransform(
                    dataset=self.dataset,
                    transform=tvt2.Compose(self.tr.train),
                ),
                classes=self.classes,
                drop_prefix=self.args.data,
            ),
            indices=idxs[:idx_split],
        )

        dataset_eval = Subset(
            MultiLabelDataset(
                dataset=DatasetTransform(
                    dataset=self.dataset,
                    transform=tvt2.Compose(self.tr.eval),
                ),
                classes=self.args.data_classes,
                drop_prefix=self.args.data,
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
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.workers,
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
        self.init_train()
        self.init_eval()

    def init_train(self):
        trans_train = [
            tvt2.ToImage(),
            tvt2.RandomHorizontalFlip(p=0.25),
            tvt2.RandomVerticalFlip(p=0.25),
            RandomRotateChoice(angles=[0, 0, 0, 0, 90, 180]),
            tvt2.RandomChoice(
                [
                    # ignore aspect ratio -- distorted square image
                    tvt2.Resize(size=(self.args.size_resize, self.args.size_resize)),
                    # keep aspect ratio
                    tvt2.Resize(size=self.args.size_resize),
                ],
                p=[1 if self.args.data_crop != "center" else 0, 1],
            ),
            tvt2.RandomChoice(
                [
                    # square crop
                    tvt2.RandomCrop(size=self.args.size_crop),
                    tvt2.CenterCrop(size=self.args.size_crop),
                ],
                p=[2 if self.args.data_crop != "center" else 0, 1],
            ),
            tvt2.ToDtype(torch.float32, scale=True),
        ]

        if self.args.data_normalize:
            trans_train.append(tvt2.Normalize(self.IMG_MEAN, self.IMG_STD))

        if self.args.data_autocontrast:
            trans_train.append(tvf.autocontrast)

        if self.args.data_equalize:
            trans_train.append(tvf.equalize)

        self.train = trans_train

    def init_eval(self):

        trans_eval = [
            tvt2.ToImage(),
            tvt2.RandomChoice(
                [
                    # ignore aspect ratio -- distorted square image
                    tvt2.Resize(size=(self.args.size_resize, self.args.size_resize)),
                    # keep aspect ratio
                    tvt2.Resize(size=self.args.size_resize),
                ],
                p=[1, 1],
            ),
            # tvt2.Resize(
            #     # ignore aspect ratio
            #     size=(self.args.size_resize, self.args.size_resize),
            # ),
            tvt2.CenterCrop(size=self.args.size_crop),
            tvt2.ToDtype(torch.float32, scale=True),
        ]

        if self.args.data_normalize:
            trans_eval.append(tvt2.Normalize(self.IMG_MEAN, self.IMG_STD))

        if self.args.data_autocontrast:
            trans_eval.append(tvf.autocontrast)

        if self.args.data_equalize:
            trans_eval.append(tvf.equalize)

        self.eval = trans_eval


def main():
    """
    Somewhere we have to start. Let's start here.
    """

    # read cli options
    args = read_args()
    setup_logging(args)

    log.info("the train is departing right now...")
    for k, v in vars(args).items():
        log.info("%s=%s", k, v)

    # train
    tm_start = time.time()
    model = Trainer(args, Data(args))
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

    parser = argparse.ArgumentParser(
        description="Train a model on a images dataset (single-label)",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        default=False,
    )
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
    parser.add_argument(
        "--data",
        type=str,
        default="data/",
    )
    parser.add_argument(
        "--data-classes",
        type=str,
        default="",
    )
    parser.add_argument(
        "--data-normalize",
        "-dn",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--data-equalize",
        "-dq",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--data-autocontrast",
        "-da",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--data-crop",
        "-dc",
        type=str,
        default="random",
        choices=["random", "center"],
    )
    parser.add_argument(
        "--data-reload",
        type=int,
        default=DATA_RELOAD,
    )
    parser.add_argument(
        "--data-train-pc",
        type=int,
        default=DATA_TRAIN_PC,
    )
    parser.add_argument(
        "--size-crop",
        type=int,
        default=DATA_CROP,
    )
    parser.add_argument(
        "--size-resize",
        type=int,
        default=DATA_RESIZE,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )
    parser.add_argument(
        "--epochs-limit",
        type=int,
        default=EPOCHS_LIMIT,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
    )

    parser.add_argument(
        "--log",
        type=str,
        default=f"k{ts}.log",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() // 4,
    )

    args = parser.parse_args()

    if args.size_resize and args.size_crop > args.size_resize:
        args.size_crop = args.size_resize

    if args.data_train_pc < 0 or args.data_train_pc > 100:
        args.data_eval_pc = DATA_TRAIN_PC

    if args.data_classes:
        args.data_classes = args.data_classes.split(",")

    if not args.model_path:
        if args.save_all:
            args.model_path = f"k{ts}_{{epoch}}.pth"
        else:
            args.model_path = f"k{ts}.pth"

    return args


if __name__ == "__main__":
    main()
