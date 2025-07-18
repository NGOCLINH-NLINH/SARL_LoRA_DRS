# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from backbone.vit import vitsmall
import torch.nn.functional as F
from datasets.seq_tinyimagenet import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
from argparse import Namespace

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class MyCIFAR100(CIFAR100):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCIFAR100Vit(ContinualDataset):

    NAME = 'seq-cifar100-vit'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10

    TRANSFORM = transforms.Compose(
            [
             transforms.Resize((224, 224)),
             transforms.RandomCrop(224, padding=28),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
             ]
    )

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        super(SequentialCIFAR100Vit, self).__init__(args)

        SequentialCIFAR100Vit.N_TASKS = args.num_tasks
        SequentialCIFAR100Vit.N_CLASSES_PER_TASK = 100 // args.num_tasks
        print(f'Setting the Number of tasks to {SequentialCIFAR100Vit.N_TASKS} with {SequentialCIFAR100Vit.N_CLASSES_PER_TASK} classes each')

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                   download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = CIFAR100(base_path() + 'CIFAR100', train=False,
                                    download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.get_normalization_transform()
        ])

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                   download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100Vit.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return vitsmall(SequentialCIFAR100Vit.N_CLASSES_PER_TASK
                        * SequentialCIFAR100Vit.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(IMAGENET_MEAN, IMAGENET_STD)
        return transform
