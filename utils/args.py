# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--experiment_id', type=str, default='cl')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--backbone', type=str, default='default')
    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_NAMES)
    parser.add_argument('--mnist_seed', type=int, default=0)
    parser.add_argument('--return_index', type=bool, default=False)
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--tiny_imagenet_path', type=str, default='data')
    parser.add_argument('--model', type=str, required=True, help='Model name.', choices=get_all_models())
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--n_epochs', type=int, required=True)
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--cifar100_num_tasks', type=int, default=5)
    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--output_dir', type=str, default='experiments')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, required=True,
                        help='The batch size of the memory buffer.')


def add_gcil_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments required for GCIL-CIFAR100 Dataset.
    :param parser: the parser instance
    """
    # arguments for GCIL-CIFAR100
    parser.add_argument('--gil_seed', type=int, default=1993, help='Seed value for GIL-CIFAR task sampling')
    parser.add_argument('--pretrain', action='store_true', default=False, help='whether to use pretrain')
    parser.add_argument('--phase_class_upper', default=50, type=int, help='the maximum number of classes')
    parser.add_argument('--epoch_size', default=1000, type=int, help='Number of samples in one epoch')
    parser.add_argument('--pretrain_class_nb', default=0, type=int, help='the number of classes in first group')
    parser.add_argument('--weight_dist', default='unif', type=str, help='what type of weight distribution assigned to classes to sample (unif or longtail)')

