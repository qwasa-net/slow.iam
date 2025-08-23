import os

import torch
from torch.utils.data import Dataset


class MultiLabelDataset(Dataset):
    """
    MultiLabelDataset is a wrapper for a dataset
    that converts a path to a multi-label target.
    """

    def __init__(self, dataset, classes=None, drop_prefix=None, transform=None):
        self.dataset = dataset
        if not classes or not isinstance(classes, list):
            self.classes = dataset.classes
            self.class_to_idx = dataset.class_to_idx
        else:
            self.classes = classes
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = dataset.samples
        self.drop_prefix = drop_prefix
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item, target = self.dataset[index]
        sample = self.dataset.samples[index]
        targets = self.path_to_targets(sample[0])
        if self.transform:
            item = self.transform(item)
        return item, targets

    def path_to_targets(self, path):
        if self.drop_prefix:
            path = path[len(self.drop_prefix) :]
        dirname, filename = os.path.split(path)
        path_parts = set(filter(lambda x: not str(x).startswith("_"), dirname.split(os.sep)))
        path_parts.intersection_update(self.classes)
        path_targets = [c in path_parts for c in self.class_to_idx]
        return torch.tensor(path_targets, dtype=torch.bool)
