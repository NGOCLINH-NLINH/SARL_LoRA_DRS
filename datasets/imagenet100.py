import numpy as np
import torch
from torchvision import transforms
import pickle


class XYDataset(torch.utils.data.Dataset):
    """
    Image pre-processing
    """

    def __init__(self, x, y, **kwargs):
        self.x, self.y = x, y

        # this was to store the inverse permutation in permuted_mnist
        # so that we could 'unscramble' samples and plot them
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        if type(x) != torch.Tensor:
            # mini_imagenet
            # we assume it's a path --> load from file
            x = self.transform(x)
            y = torch.Tensor(1).fill_(y).long().squeeze()
        else:
            x = x.float() / 255.
            y = y.long()

        # for some reason mnist does better \in [0,1] than [-1, 1]
        if self.source == 'mnist':
            return x, y
        else:
            return (x - 0.5) * 2, y


class CLDataLoader(object):
    """
    Create data loader for the given task dataset
    """

    def __init__(self, datasets_per_task, args, train=True):
        bs = args.batch_size if train else 256

        self.datasets = datasets_per_task
        self.loaders = [
            torch.utils.data.DataLoader(x, batch_size=bs, shuffle=True, drop_last=train, num_workers=0)
            for x in self.datasets]

    def __getitem__(self, idx):
        return self.loaders[idx]

    def __len__(self):
        return len(self.loaders)


def get_miniimagenet(args, get_val=False):
    """
    Import mini-imagenet dataset and split it into multiple tasks with disjoint set of classes
    Implementation is based on the one provided by:
        Aljundi, Rahaf, et al. "Online continual learning with maximally interfered retrieval."
        arXiv preprint arXiv:1908.04742 (2019).
    :param args: Arguments for model/data configuration
    :param get_val: Get validation set for grid search
    :return: Train, test and validation data loaders
    """
    args.n_classes = 100
    args.n_classes_per_task = 5
    args.input_size = (3, 84, 84)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    for i in ['train', 'test', 'val']:
        file = open("data/mini_imagenet/mini-imagenet-cache-" + i + ".pkl", "rb")
        file_data = pickle.load(file)
        data = file_data["image_data"]
        if i == 'train':
            main_data = data.reshape([64, 600, 84, 84, 3])
        else:
            app_data = data.reshape([(20 if i == 'test' else 16), 600, 84, 84, 3])
            main_data = np.append(main_data, app_data, axis=0)
    all_data = main_data.reshape((60000, 84, 84, 3))
    all_label = np.array([[i] * 600 for i in range(100)]).flatten()

    train_ds, test_ds = [], []
    current_train, current_test = None, None

    cat = lambda x, y: np.concatenate((x, y), axis=0)

    for i in range(args.n_classes):
        class_indices = np.argwhere(all_label == i).reshape(-1)
        class_data = all_data[class_indices]
        class_label = all_label[class_indices]
        split = int(0.8 * class_data.shape[0])

        data_train, data_test = class_data[:split], class_data[split:]
        label_train, label_test = class_label[:split], class_label[split:]

        if current_train is None:
            current_train, current_test = (data_train, label_train), (data_test, label_test)
        else:
            current_train = cat(current_train[0], data_train), cat(current_train[1], label_train)
            current_test = cat(current_test[0], data_test), cat(current_test[1], label_test)

        if i % args.n_classes_per_task == (args.n_classes_per_task - 1):
            train_ds += [current_train]
            test_ds += [current_test]
            current_train, current_test = None, None

    # build masks
    masks = []
    task_ids = [None for _ in range(20)]
    for task, task_data in enumerate(train_ds):
        labels = np.unique(task_data[1])  # task_data[1].unique().long()
        assert labels.shape[0] == args.n_classes_per_task
        mask = torch.zeros(args.n_classes).to(args.device)
        mask[labels] = 1
        masks += [mask]
        task_ids[task] = labels

    task_ids = torch.from_numpy(np.stack(task_ids)).to(args.device).long()

    test_ds = map(lambda x, y: XYDataset(x[0], x[1], **{'source': 'mini_imagenet', 'mask': y, 'task_ids': task_ids,
                                                        'transform': transform}), test_ds, masks)
    if get_val:
        train_ds, val_ds = make_valid_from_train(train_ds)
        val_ds = map(lambda x, y: XYDataset(x[0], x[1], **{'source': 'mini_imagenet', 'mask': y, 'task_ids': task_ids,
                                                           'transform': transform}), val_ds, masks)
    else:
        val_ds = test_ds
    train_ds = map(lambda x, y: XYDataset(x[0], x[1], **{'source': 'mini_imagenet', 'mask': y, 'task_ids': task_ids,
                                                         'transform': transform}), train_ds, masks)

    return train_ds, test_ds, val_ds


def make_valid_from_train(dataset, cut=0.9):
    """
    Split training data to get validation set
    :param dataset: Training dataset
    :param cut: Percentage of dataset to be kept for training purpose
    """
    tr_ds, val_ds = [], []
    for task_ds in dataset:
        x_t, y_t = task_ds

        # shuffle before splitting
        perm = torch.randperm(len(x_t))
        x_t, y_t = x_t[perm], y_t[perm]

        split = int(len(x_t) * cut)
        x_tr, y_tr = x_t[:split], y_t[:split]
        x_val, y_val = x_t[split:], y_t[split:]

        tr_ds += [(x_tr, y_tr)]
        val_ds += [(x_val, y_val)]

    return tr_ds, val_ds