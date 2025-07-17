from torchvision import datasets
from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Tuple


class CIFAR100SC(datasets.CIFAR100):
    """CIFAR100 dataset, with superclass Info
    """

    def __init__(self, **kwargs):
        super(CIFAR100SC, self).__init__(**kwargs)

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []
        self.coarse_target = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

                self.coarse_target.extend(entry['coarse_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])

        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.super_classes = data['coarse_label_names']
            self.super_classes = data['coarse_label_names']
        self.super_class_to_idx = {_class: i for i, _class in enumerate(self.super_classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, coarse_target = self.data[index], self.targets[index], self.coarse_target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, coarse_target

data = CIFAR100SC(root='data', train=False, download=True)

data.super_class_to_idx

groups = {}

for target, cat in zip(data.targets, data.coarse_target):
    if cat in groups:
        if not target in groups[cat]:
            groups[cat].append(target)
    else:
        groups[cat] = [target]


pos_groups = {}

for group_id in range(len(groups)):
    for class_id in groups[group_id]:
        pos_groups[class_id] = groups[group_id]

pos_groups_ordered = {}
for class_id in range(len(pos_groups)):
    pos_groups_ordered[class_id] = pos_groups[class_id]

with open(r'/home/fahad.sarfraz/workspace/SARL/data/cifar-100-python/meta', 'rb') as infile:
    data = pickle.load(infile, encoding='latin1')

labels = data['fine_label_names']

labels_dict = {}

for i in range(100):
    labels_dict[i] = labels[i]
