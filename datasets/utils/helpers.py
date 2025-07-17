import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import os
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)

transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

def get_tinyimg_transform():
    MEAN, STD = (0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821)
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(64, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])
    return TRANSFORM


def get_imagenet_transform():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


def group_classes(dataset):
    """
    Groups images in the dataset by their labels.
    :param dataset: A dataset of (image, label) pairs.
    :return: A defaultdict(list) with labels as keys and lists of images as values.
    """
    grouped_images = defaultdict(list)
    for img, label in dataset:
        grouped_images[label].append(img)

    return grouped_images


def save_task_images_to_memmap(grouped_images, output_dir, n_classes_per_task=2, train=True):
    """
    Saves images for each task to separate numpy.memmap files.
    :param grouped_images: Dictionary with class labels as keys and lists of PIL images as values.
    :param output_dir: Directory to save the memmap files.
    :param n_classes_per_task: Number of classes per task.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Organize images by tasks
    task_images = defaultdict(list)
    for class_label, images in grouped_images.items():
        task_idx = class_label // n_classes_per_task
        task_images[task_idx].extend(images)

    for task_idx, images in task_images.items():
        # Convert images to numpy arrays and flatten the list
        images_np = np.array([np.array(img) for img in images])
        labels = [class_label for class_label, imgs in grouped_images.items() for _ in range(len(imgs)) if
                  class_label // n_classes_per_task == task_idx]
        # Create and save the memmap file for images
        if train:
            memmap_path = os.path.join(output_dir, f'task_{task_idx}_train_images.memmap')
        else:
            memmap_path = os.path.join(output_dir, f'task_{task_idx}_test_images.memmap')
        images_memmap = np.memmap(memmap_path, dtype=np.uint8, mode='w+', shape=images_np.shape)
        images_memmap[:] = images_np[:]
        del images_memmap  # Flush to disk

        # Save labels to a numpy array file
        if train:
            labels_path = os.path.join(output_dir, f'task_{task_idx}_train_labels.npy')
        else:
            labels_path = os.path.join(output_dir, f'task_{task_idx}_test_labels.npy')
        np.save(labels_path, np.array(labels, dtype=np.int32))


def create_task_dataloaders(output_dir, n_tasks=5, batch_size=32, transform=None, shuffle=True, train=True, image_shape=(32, 32, 3)):
    dataloaders = []
    for task_idx in range(n_tasks):
        if train:
            memmap_path = os.path.join(output_dir, f'task_{task_idx}_train_images.memmap')
            labels_path = os.path.join(output_dir, f'task_{task_idx}_train_labels.npy')
        else:
            memmap_path = os.path.join(output_dir, f'task_{task_idx}_test_images.memmap')
            labels_path = os.path.join(output_dir, f'task_{task_idx}_test_labels.npy')
        dataset = CustomDataset(memmap_path, labels_path, transform=transform, image_shape=image_shape)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        dataloaders.append(dataloader)
    return dataloaders


class CustomDataset(Dataset):
    """Custom Dataset class compatible with numpy.memmap."""

    def __init__(self, memmap_path, labels_path, transform=None, image_shape=(32, 32, 3)):
        """
        Initializes the dataset with memory-mapped images.
        :param memmap_path: Path to the memory-mapped file containing images.
        :param labels: List of labels corresponding to each image.
        :param transform: Transformations to be applied to images.
        :param image_shape: Shape of each image in the memmap.
        """
        self.labels = np.load(labels_path)
        self.memmap = np.memmap(memmap_path, dtype=np.uint8, mode='r', shape=(len(self.labels), *image_shape))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.memmap[idx]
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)
        return image, label


def create_seq_dataloaders(grouped_images, n_classes_per_task, batch_size=32, transform=None, shuffle=True, train=True, args=None, save_mem_map=True):
    """
    Creates sequential dataloaders for tasks.
    :param grouped_images: List of images grouped by class.
    :param n_classes_per_task: Number of classes per task.
    :param batch_size: Batch size for dataloader.
    :param transform: Transformations to be applied to images.
    :param anti_images: Additional images to be included in each task.
    :param anti_label: Label for the additional images.
    :return: List of dataloaders, each corresponding to a task.
    """
    num_classes = len(grouped_images)  # Number of unique classes
    if args.dataset == 'tinyimg':
        image_shape = (64, 64, 3)
    else:
        image_shape = (32, 32, 3)

    n_tasks = int(np.ceil(num_classes / n_classes_per_task))
    if save_mem_map:
        save_task_images_to_memmap(grouped_images, args.data_root, n_classes_per_task=n_classes_per_task, train=train)
    dataloaders = create_task_dataloaders(args.data_root, n_tasks=n_tasks, batch_size=batch_size, transform=transform,
                                          shuffle=shuffle, train=train, image_shape=image_shape)
    return dataloaders


def create_sequential_dataloaders(train_dataset, test_dataset, n_classes, total_classes, batch_size):
    train_class_indices = {}
    test_class_indices = {}

    for i in range(total_classes):
        train_class_indices[i] = [idx for idx, label in enumerate(train_dataset.targets) if label == i]
        test_class_indices[i] = [idx for idx, label in enumerate(test_dataset.targets) if label == i]

    train_dataloaders = []
    test_dataloaders = []

    for i in range(0, total_classes, n_classes):
        classes_subset = list(range(i, i + n_classes))
        train_indices = [idx for c in classes_subset for idx in train_class_indices[c] if idx < len(train_dataset)]
        test_indices = [idx for c in classes_subset for idx in test_class_indices[c] if idx < len(test_dataset)]

        train_subset = data.Subset(train_dataset, train_indices)
        test_subset = data.Subset(test_dataset, test_indices)

        train_loader = data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = data.DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        train_dataloaders.append(train_loader)
        test_dataloaders.append(test_loader)

    return train_dataloaders, test_dataloaders


def create_dataloaders(train_data, test_data=None, n_classes_per_task=2, batch_size=32, transform=None,
                       anti_images=None, args=None, save_mem_map=True):
    """
    Creates dataloaders for training and testing datasets.

    Parameters:
    train_data: List of grouped training data.
    test_data: List of grouped testing data (optional).
    batch_size: Size of each data batch.
    transform: Transformations to apply to the data (optional).

    Returns:
    A tuple of (train_dataloaders, test_dataloaders).
    """
    train_dataloaders = create_seq_dataloaders(train_data, n_classes_per_task=n_classes_per_task, batch_size=batch_size,
                                               transform=transform, shuffle=True, train=True, args=args, save_mem_map=save_mem_map)
    test_dataloaders = create_seq_dataloaders(test_data, n_classes_per_task=n_classes_per_task, batch_size=batch_size,
                                              transform=transform_base, shuffle=False, train=False, args=args, save_mem_map=save_mem_map) if test_data else []

    return train_dataloaders, test_dataloaders

