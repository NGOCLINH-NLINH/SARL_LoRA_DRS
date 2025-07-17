# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torchvision.transforms as transforms
from backbone.ResNet18_imagenet import resnet18
import torch.nn.functional as F
import os
from datasets.utils.continual_dataset import ContinualDataset
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import get_previous_train_loader
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from typing import Tuple
from torchvision import datasets

DROP_LAST = True


class ImageNet100(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, split="train"):
        super().__init__(root=root, transform=transform, target_transform=target_transform)
        self.not_aug_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

        self.targets = np.array([sample[1] for sample in self.samples])
        self.data = np.array([sample[0] for sample in self.samples])
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

    def __getitem__(self, index):
        """Overwrite getitem to handle tasks and custom transforms."""
        # path, target = self.samples[index]
        path, target = self.data[index], self.targets[index]
        img = self.loader(path)
        original_img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.split == "train":
            not_aug_img = self.not_aug_transform(original_img)
            return img, target, not_aug_img
        else:
            return img, target


class SequentialImagenet100(ContinualDataset):

    NAME = 'seq-imagenet100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10

    TRANSFORM = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                             setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
        """
        Divides the dataset into tasks.
        :param train_dataset: train dataset
        :param test_dataset: test dataset
        :param setting: continual learning setting
        :return: train and test loaders
        """
        train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
                                    np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

        test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i,
                                   np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

        train_dataset.data = train_dataset.data[train_mask]
        test_dataset.data = test_dataset.data[test_mask]

        train_dataset.targets = np.array(train_dataset.targets)[train_mask]
        test_dataset.targets = np.array(test_dataset.targets)[test_mask]

        train_dataset.samples = [sample for n, sample in enumerate(train_dataset.samples) if train_mask[n]]
        test_dataset.samples = [sample for n, sample in enumerate(test_dataset.samples) if test_mask[n]]

        train_loader = DataLoader(train_dataset,
                                  batch_size=setting.args.batch_size, shuffle=True, num_workers=0, drop_last=DROP_LAST)
        test_loader = DataLoader(test_dataset,
                                 batch_size=setting.args.batch_size, shuffle=False, num_workers=0)
        setting.test_loaders.append(test_loader)
        setting.train_loader = train_loader

        print(np.unique(train_dataset.targets))
        print(len(train_dataset.targets), len(train_dataset.data), len(train_dataset.samples))
        print(len(test_dataset.targets), len(test_dataset.data), len(test_dataset.samples))

        setting.i += setting.N_CLASSES_PER_TASK
        return train_loader, test_loader

from torch.utils.data import Dataset
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
import os
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from datasets.transforms.denormalization import DeNormalize


class TinyImagenet(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from google_drive_downloader import GoogleDriveDownloader as gdd

                # https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view
                print('Downloading dataset')
                gdd.download_file_from_google_drive(
                    file_id='1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj',

                    dest_path=os.path.join(root, 'tiny-imagenet-processed.zip'),
                    unzip=True)

        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.targets = np.concatenate(np.array(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class MyTinyImagenet(TinyImagenet):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        super(MyTinyImagenet, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target,  not_aug_img


class SequentialTinyImagenet(ContinualDataset):

    NAME = 'seq-tinyimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(64, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4802, 0.4480, 0.3975),
                                  (0.2770, 0.2691, 0.2821))])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = ImageNet100(
            root=os.path.join(str(self.args.data_root), "train"),
            transform=transform,
            split="train"
        )

        test_dataset = ImageNet100(
            root=os.path.join(str(self.args.data_root), "val"),
            transform=test_transform,
            split="test",
        )

        train, test = self.store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyTinyImagenet(os.path.join(self.args.tiny_imagenet_path, 'TINYIMG'),
                                 train=True, download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = TinyImagenet(os.path.join(self.args.tiny_imagenet_path, 'TINYIMG'),
                        train=False, download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_denormalization_transform()])

        train_dataset = MyTinyImagenet(os.path.join(self.args.tiny_imagenet_path, 'TINYIMG'),
                            train=True, download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    @staticmethod
    def get_backbone():
        return resnet18(SequentialImagenet100.N_CLASSES_PER_TASK
                        * SequentialImagenet100.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))
        return transform


if __name__ == "__main__":
    from argparse import Namespace
    import torchvision.datasets as datasets

    args = Namespace(
        data_root="/home/ubuntu/as_experiments/SARL/data/imagenet100",
        n_classes_per_task=10,
        n_classes=100,
        batch_size=32,
    )

    dataset = SequentialImagenet100(args)

    for i in range(dataset.N_TASKS):
        train_loader, test_loader = dataset.get_data_loaders()
        for data, label, non_aug_data in train_loader:
            print(data.shape)
