import torch
torch.set_num_threads(4)
from copy import deepcopy
from torch import nn
from torch.optim import SGD
from backbone.SparseResNet18 import sparse_resnet18
import torch.nn.functional as F
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from torch.utils.data import DataLoader


num_classes_dict = {
    'seq-cifar10': 10,
    'seq-cifar100': 100,
    'seq-tinyimg': 200,
    'gcil-cifar100': 100,
    'seq-imagenet100': 100,
}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Complementary Learning Systems Based Experience Replay')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # Consistency Regularization Weight
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--op_weight', type=float, default=0.1)
    parser.add_argument('--sim_thresh', type=float, default=0.80)
    parser.add_argument('--sm_weight', type=float, default=0.01)
    # Sparsity param
    parser.add_argument('--apply_kw', nargs='*', type=int, default=[1, 1, 1, 1])
    parser.add_argument('--kw', type=float, nargs='*', default=[0.9, 0.9, 0.9, 0.9])
    parser.add_argument('--kw_relu', type=int, default=1)
    parser.add_argument('--kw_local', type=int, default=1)
    parser.add_argument('--num_feats', type=int, default=512)
    # Experimental Args
    parser.add_argument('--save_interim', type=int, default=1)
    parser.add_argument('--warmup_iters', type=int, default=10)
    return parser


# =============================================================================
# Mean-ER
# =============================================================================
class SARLOnline(ContinualModel):
    NAME = 'sarl_online'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(SARLOnline, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        # Initialize plastic and stable model
        self.net = sparse_resnet18(
            nclasses=num_classes_dict[args.dataset],
            kw_percent_on=args.kw,
            local=args.kw_local,
            relu=args.kw_relu,
            apply_kw=args.apply_kw
        ).to(self.device)

        self.net_old = None
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)

        # set regularization weight
        self.alpha = args.alpha
        self.consistency_loss = nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0
        self.lst_models = ['net']

        # init Class Prototypes
        self.op = torch.zeros(num_classes_dict[args.dataset], args.num_feats).to(self.device)
        self.running_op = torch.zeros(num_classes_dict[args.dataset], args.num_feats).to(self.device)
        self.running_feat_sum = torch.zeros(num_classes_dict[args.dataset], args.num_feats).to(self.device)
        self.running_sample_counts = torch.zeros(num_classes_dict[args.dataset]).to(self.device)

        self.learned_classes = []
        self.new_classes = []
        self.flag = True
        self.eval_prototypes = True
        self.pos_groups = {}
        self.dist_mat = torch.zeros(num_classes_dict[args.dataset], num_classes_dict[args.dataset]).to(self.device)
        self.iter = 0

    def observe(self, inputs, labels, not_aug_inputs):

        self.iter += 1
        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        self.net.train()

        loss = 0
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buff_out, buff_feats = self.net(buf_inputs, return_activations=True)

            if isinstance(buff_feats, dict):
                buff_feats = buff_feats['feat']
            reg_loss = self.args.alpha * F.mse_loss(buff_out, buf_logits)

            buff_ce_loss = self.loss(buff_out, buf_labels)
            loss += reg_loss + buff_ce_loss

            # Regularization loss on Class Prototypes
            if self.current_task > 0:
                buff_feats = F.normalize(buff_feats)
                dist = 0
                for class_label in torch.unique(buf_labels):
                    if class_label in self.learned_classes:
                        image_class_mask = (buf_labels == class_label)
                        mean_feat = buff_feats[image_class_mask].mean(axis=0)
                        dist += F.mse_loss(mean_feat, self.op[class_label])
                loss += self.args.op_weight * dist

            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/reg_loss', reg_loss.item(), self.iteration)
                self.writer.add_scalar(f'Task {self.current_task}/buff_ce_loss', buff_ce_loss.item(), self.iteration)

        outputs, activations = self.net(inputs, return_activations=True)

        # update new classes
        unique_labels = labels.unique()
        unique_labels = [label.item() for label in unique_labels]
        for class_label in unique_labels:
            if class_label not in self.learned_classes:
                if class_label not in self.new_classes:
                    self.new_classes.append(class_label)

        # Update running mean
        if self.iter > self.args.warmup_iters:

            # Update running mean
            unique_labels = labels.unique()
            feat = activations['feat']
            feat = F.normalize(feat)

            for class_label in unique_labels:
                self.running_feat_sum[class_label] += feat[labels == class_label].sum(dim=0).detach()
                self.running_sample_counts[class_label] += (labels == class_label).sum().detach()

            if self.current_task > 0:

                if self.eval_prototypes:
                    self.create_groups(self.new_classes)

                # Apply functional distillation
                outputs_old = self.net_old(inputs)
                loss += self.args.beta * F.mse_loss(outputs, outputs_old)

                new_labels = [i.item() for i in torch.unique(labels) if i not in self.learned_classes]
                all_labels = self.learned_classes + new_labels

                # Accumulate the class prototypes
                class_prot = {}
                for ref_class_label in self.learned_classes:
                    class_prot[ref_class_label] = self.op[ref_class_label]
                for class_label in new_labels:
                    class_prot[class_label] = feat[labels == class_label].mean(dim=0)

                l_cont = 0
                for class_label in new_labels:
                    pos_dist = 0
                    neg_dist = 0
                    for ref_class_label in all_labels:
                        if class_label != ref_class_label:
                            if ref_class_label in self.pos_groups[class_label]:
                                pos_dist += F.mse_loss(class_prot[class_label], class_prot[ref_class_label])
                            else:
                                neg_dist += F.mse_loss(class_prot[class_label], class_prot[ref_class_label])

                    if neg_dist > 0:
                        l_cont += pos_dist/neg_dist

                loss += self.args.sm_weight * l_cont
                if hasattr(self, 'writer'):
                    self.writer.add_scalar(f'Task {self.current_task}/contrastive_loss', l_cont.item(), self.iteration)

        ce_loss = self.loss(outputs, labels)
        loss += ce_loss

        if torch.isnan(loss):
            raise ValueError('NAN Loss')

        # Log values
        if hasattr(self, 'writer'):
            self.writer.add_scalar(f'Task {self.current_task}/ce_loss', ce_loss.item(), self.iteration)
            self.writer.add_scalar(f'Task {self.current_task}/loss', loss.item(), self.iteration)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels[:real_batch_size],
            logits=outputs.data,
        )

        return loss.item()

    def create_groups(self, new_labels) -> None:

        print('!' * 30)
        print('Evaluating Prototypes for the New Classes', new_labels)
        # Take average feats
        for class_label in new_labels:
            self.running_op[class_label] = self.running_feat_sum[class_label] / self.running_sample_counts[class_label]

        # Evaluate the distances
        all_labels = self.learned_classes + new_labels

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        # dist_mat = torch.zeros(len(all_labels), len(all_labels))
        for class_label in new_labels:
            for ref_class_label in new_labels:
                self.dist_mat[class_label, ref_class_label] = cos(self.running_op[class_label], self.running_op[ref_class_label])
            for ref_class_label in self.learned_classes:
                self.dist_mat[class_label, ref_class_label] = cos(self.running_op[class_label], self.op[ref_class_label])

        print(self.dist_mat)

        print('*' * 30)
        print('Positive Groups')
        for class_label in new_labels:
            pos_group = self.dist_mat[class_label] > self.args.sim_thresh
            self.pos_groups[class_label] = [i for i in all_labels if pos_group[i]]
            print(f'{class_label}:', self.pos_groups[class_label])
        print('*' * 30)
        self.eval_prototypes = False

    def end_task(self, dataset) -> None:
        self.iter = 0

        self.eval_prototypes = True
        self.flag = True
        self.current_task += 1
        self.net.eval()

        # Save old model
        self.net_old = deepcopy(self.net)
        self.net_old.eval()

        # =====================================
        # Buffer Pass
        # =====================================
        buf_inputs, buf_labels, buf_logits = self.buffer.get_all_data(transform=self.transform)
        buf_idx = torch.arange(0, len(buf_labels)).to(buf_labels.device)

        buff_dataset = torch.utils.data.TensorDataset(buf_inputs, buf_labels, buf_logits, buf_idx)
        buff_data_loader = DataLoader(buff_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0)

        self.net.train()
        for data, label, logits, index in buff_data_loader:
            out_net = self.net(data)

        # =====================================
        # Update Class Prototypes
        # =====================================
        for class_label in self.new_classes:
            if class_label not in self.learned_classes:
                self.learned_classes.append(class_label)
            self.op[class_label] = self.running_feat_sum[class_label] / self.running_sample_counts[class_label]

        print(self.new_classes)
        self.new_classes = []
