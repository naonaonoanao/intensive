from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SignDataset(Dataset):
    def __init__(self, paths: List[Path], transform=None):
        self.paths = paths
        self.transform = transform # если есть аугментации
        labels = sorted(set(str(x).split('/')[-2] for x in paths))
        self.one_hot_encoding = {label: i for i, label in enumerate(labels)}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        def resize_with_padding(image, target_size):
            h, w, c = image.shape
            target_h, target_w = target_size

            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            padded = np.zeros((target_h, target_w, c), dtype=image.dtype)

            top = (target_h - new_h) // 2
            left = (target_w - new_w) // 2
            padded[top:top + new_h, left:left + new_w] = resized

            return padded

        # Чтение изображения
        image = cv2.imread(str(self.paths[idx]))
        label = str(self.paths[idx]).split('/')[-2]

        image = resize_with_padding(image, (200, 200))
        image = np.transpose(image, (2, 0, 1))

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image).float(), torch.tensor(self.one_hot_encoding[label])


def get_sign_dataloader(
        path_train, path_val, batch_size, shuffle=True, num_workers=1,
    ):
    train_dataset = SignDataset(paths=[*Path(path_train).rglob('*.jpg')])
    val_dataset = SignDataset(paths=[*Path(path_val).rglob('*.jpg')])

    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers
    }
    return DataLoader(train_dataset, **loader_args), DataLoader(val_dataset, **loader_args)


def get_sign_test_dataloader(
        path_test, batch_size, num_workers=1,
    ):
    test_dataset = SignDataset(paths=[*Path(path_test).rglob('*.jpg')])

    loader_args = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers
    }
    return DataLoader(test_dataset, **loader_args)
