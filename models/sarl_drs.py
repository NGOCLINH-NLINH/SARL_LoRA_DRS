import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from copy import deepcopy
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from backbone.MNISTMLP import SparseMNISTMLP
from backbone.SparseResNet18 import sparse_resnet18
from models.utils.losses import SupConLoss
from models.utils.pos_groups import class_dict, pos_groups
from backbone.vit_sparse_lora import vitsmall_sparse_lora
from backbone.utils.attention_lora import Attention_LoRA
import timm


num_classes_dict = {
    'seq-cifar10': 10,
    'seq-cifar100': 100,
    'seq-tinyimg': 200,
    'gcil-cifar100': 100,
    'seq-imagenet100': 100,
    'seq-cifar100-vit': 100,
}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Semantic Aware Representation Learning')
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
    # parser.add_argument('--apply_kw', nargs='*', type=int, default=[1, 1, 1, 1])
    parser.add_argument('--kw', type=float, default=0.9)
    # parser.add_argument('--kw_relu', type=int, default=1)
    # parser.add_argument('--kw_local', type=int, default=1)
    parser.add_argument('--num_feats', type=int, default=384)
    # Experimental Args
    parser.add_argument('--save_interim', type=int, default=1)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--use_lr_scheduler', type=int, default=1)
    parser.add_argument('--lr_steps', type=int, nargs='*', default=[70, 90])

    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--drs_variance', type=float, default=0.99)
    return parser


# =============================================================================
# Mean-ER
# =============================================================================
class SARLDRS(ContinualModel):
    NAME = 'sarl_drs'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(SARLDRS, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        print("Initializing new SparseVitLoRA model...")
        # Initialize plastic and stable model
        if 'mnist' in self.args.dataset:
            self.net = SparseMNISTMLP(28 * 28, 10, kw_percent_on=args.kw).to(self.device)
        else:
            self.net = vitsmall_sparse_lora(
            nclasses=num_classes_dict[args.dataset],
            kw_percent_on=args.kw,
            lora_r=args.lora_r,
            num_tasks=args.num_tasks
        ).to(self.device)

        pretrained_model_name = 'vit_base_patch16_224_in21k'
        print(f"Loading pre-trained weights for '{pretrained_model_name}' using timm...")

        pretrained_model = timm.create_model(pretrained_model_name, pretrained=True)
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict)

        print("Successfully loaded pre-trained backbone weights.")
        self.w0 = {name: p.clone().detach() for name, p in self.net.named_parameters()}

        self.drs_projections = {}
        self.drs_setup_done = False

        self.net_old = None

        # set regularization weight
        self.alpha = args.alpha

        self.consistency_loss = nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0
        self.lst_models = ['net']

        self.get_optimizer()

        # init Object Prototypes
        self.op = torch.zeros(num_classes_dict[args.dataset], args.num_feats).to(self.device)
        self.op_sum = torch.zeros(num_classes_dict[args.dataset], args.num_feats).to(self.device)
        self.sample_counts = torch.zeros(num_classes_dict[args.dataset]).to(self.device)

        self.running_op = torch.zeros(num_classes_dict[args.dataset], args.num_feats).to(self.device)
        self.running_sample_counts = torch.zeros(num_classes_dict[args.dataset]).to(self.device)

        self.learned_classes = []
        self.flag = True
        self.eval_prototypes = True
        self.pos_groups = {}
        self.dist_mat = torch.zeros(num_classes_dict[args.dataset], num_classes_dict[args.dataset]).to(self.device)
        self.class_dict = class_dict[args.dataset]

    def _calculate_drs(self):
        print("Calculating Drift-Resistant Space for task {}...".format(self.current_task))
        self.net.eval()

        print("Fetching data from buffer to compute DRS...")
        buf_inputs, buf_labels, _ = self.buffer.get_all_data()

        if len(buf_inputs) == 0:
            print("Warning: Buffer is empty. Cannot compute DRS. Skipping...")
            self.drs_setup_done = True
            self.net.train()
            return

        buffer_dataset = torch.utils.data.TensorDataset(buf_inputs, buf_labels)
        temp_loader = DataLoader(buffer_dataset, batch_size=self.args.batch_size, shuffle=False)

        temp_net = deepcopy(self.net)
        temp_net_dict = temp_net.state_dict()

        with torch.no_grad():
            for name, module in temp_net.named_modules():
                if isinstance(module, Attention_LoRA):
                    w_qkv_tilde = self.w0[f'{name}.qkv.weight'].clone()
                    for task_idx in range(self.current_task):
                        w_qkv_tilde -= module.get_lora_subtraction_matrix(task_idx)
                    temp_net_dict[f'{name}.qkv.weight'] = w_qkv_tilde
        temp_net.load_state_dict(temp_net_dict)

        features = {}
        hooks = []

        def hook_fn(module, input, output):
            features[module.name] = input[0].detach()

        for name, module in temp_net.named_modules():
            if isinstance(module, Attention_LoRA):
                module.name = name
                hooks.append(module.register_forward_pre_hook(hook_fn))

        with torch.no_grad():
            for data_batch in temp_loader:
                inputs = data_batch[0]
                temp_net(inputs.to(self.device))

        for hook in hooks:
            hook.remove()

        with torch.no_grad():
            for name, feat_batch in features.items():
                B, N, C = feat_batch.shape
                X_tilde = feat_batch.view(-1, C)

                covariance = (X_tilde.T @ X_tilde) / X_tilde.shape[0]
                eigvals, eigvecs = torch.linalg.eigh(covariance)

                sorted_indices = torch.argsort(eigvals, descending=True)
                sorted_eigvecs = eigvecs[:, sorted_indices]

                cumsum_eigvals = torch.cumsum(eigvals[sorted_indices], dim=0)
                total_var = cumsum_eigvals[-1]
                k = torch.where(cumsum_eigvals / total_var >= 0.99)[0][0]

                P_t = sorted_eigvecs[:, :k + 1]
                self.drs_projections[f"{name}.qkv.weight"] = P_t
                print(f"Layer {name}: DRS subspace dimension = {P_t.shape[1]}")

        self.drs_setup_done = True
        self.net.train()

    def observe(self, inputs, labels, not_aug_inputs):
        if self.current_task > 0 and not self.drs_setup_done:
            self._calculate_drs()

        for module in self.net.modules():
            if hasattr(module, 'set_task'):
                module.set_task(self.current_task)

        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        self.net.train()
        loss = 0
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buff_out, buff_activations = self.net(buf_inputs, return_activations=True)
            buff_feats = buff_activations['feat']
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

        if self.epoch > self.args.warmup_epochs and self.current_task > 0:

            outputs_old = self.net_old(inputs)
            loss += self.args.beta * F.mse_loss(outputs, outputs_old)

            new_labels = [i.item() for i in torch.unique(labels) if i not in self.learned_classes]
            all_labels = self.learned_classes + new_labels
            feats = activations['feat']
            feats = F.normalize(feats)

            # Accumulate the class prototypes
            class_prot = {}
            for ref_class_label in self.learned_classes:
                class_prot[ref_class_label] = self.op[ref_class_label]
            for class_label in new_labels:
                class_prot[class_label] = feats[labels == class_label].mean(dim=0)

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

        if self.current_task > 0:
            with torch.no_grad():
                for name, param in self.net.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        if 'lora' in name:
                            layer_name = name.split('.lora_')[0]
                            proj_name = f"{layer_name}.qkv.weight"

                            if proj_name in self.drs_projections:
                                P_t = self.drs_projections[proj_name]

                                original_shape = param.grad.data.shape
                                grad_vec = param.grad.data.view(-1)
                                projected_grad = P_t @ (P_t.T @ grad_vec)
                                param.grad.data.copy_(projected_grad.view(original_shape))

        self.opt.step()

        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels[:real_batch_size],
            logits=outputs.data,
        )

        return loss.item()

    def end_epoch(self, dataset, epoch) -> None:

        if self.scheduler is not None:
            self.scheduler.step()

        self.flag = True
        self.net.eval()

        # Calculate the Class Prototypes and Covariance Matrices using Working Model
        if self.epoch >= self.args.warmup_epochs and self.eval_prototypes and self.current_task > 0:
            print('!' * 30)
            print('Evaluating Prototypes for the New Classes')
            # Calculate CLass Prototypes
            X = []
            Y = []
            for data in dataset.train_loader:
                inputs, labels, not_aug_inputs = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # print(input.shape)
                outputs, activations = self.net(inputs, return_activations=True)
                feat = activations['feat']

                # Normalize Features
                feat = F.normalize(feat)  # Is it needed

                X.append(feat.detach().cpu().numpy())
                Y.append(labels.cpu().numpy())

                unique_labels = labels.unique()
                for class_label in unique_labels:
                    self.running_op[class_label] += feat[labels == class_label].sum(dim=0).detach()
                    self.running_sample_counts[class_label] += (labels == class_label).sum().detach()

            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)

            # Take average feats
            for class_label in np.unique(Y):
                self.running_op[class_label] = self.running_op[class_label] / self.running_sample_counts[class_label]

            # Calculate Covariance Matrix
            class_mean_set = []
            for class_label in np.unique(Y):

                image_class_mask = (Y == class_label)
                class_mean_set.append(np.mean(X[image_class_mask], axis=0))

            # Evaluate the distances
            new_labels = [i for i in np.unique(Y) if i not in self.learned_classes]
            all_labels = self.learned_classes + new_labels

            cos = nn.CosineSimilarity(dim=0, eps=1e-6)

            # dist_mat = torch.zeros(len(all_labels), len(all_labels))
            for class_label in new_labels:
                for ref_class_label in new_labels:
                    self.dist_mat[class_label, ref_class_label] = cos(self.running_op[class_label], self.running_op[ref_class_label])
                for ref_class_label in self.learned_classes:
                    self.dist_mat[class_label, ref_class_label] = cos(self.running_op[class_label], self.op[ref_class_label])

            print('*' * 30)
            print('Positive Groups')
            for class_label in new_labels:
                pos_group = self.dist_mat[class_label] > self.args.sim_thresh
                self.pos_groups[class_label] = [i for i in all_labels if pos_group[i]]
                if self.args.dataset not in ['seq-tinyimg', 'gcil-cifar100']:
                    print(f'{self.class_dict[class_label]}: ' + ', '.join([self.class_dict[i] for i in self.pos_groups[class_label]]))
                else:
                    print(f'{class_label}:', self.pos_groups[class_label])
            print('*' * 30)

            self.eval_prototypes = False

    def end_task(self, dataset) -> None:
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
        # Calculate CLass Prototypes
        # =====================================
        self.net.eval()
        X = []
        Y = []
        for data in dataset.train_loader:
            inputs, labels, not_aug_inputs = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # print(input.shape)
            outputs, activations = self.net(inputs, return_activations=True)
            feat = activations['feat']

            # Normalize Features
            feat = F.normalize(feat)  # Is it needed

            X.append(feat.detach().cpu().numpy())
            Y.append(labels.cpu().numpy())

            unique_labels = labels.unique()
            for class_label in unique_labels:
                self.op_sum[class_label] += feat[labels == class_label].sum(dim=0).detach()
                self.sample_counts[class_label] += (labels == class_label).sum().detach()

        X = np.concatenate(X)
        Y = np.concatenate(Y)

        # Take average feats
        for class_label in np.unique(Y):
            if class_label not in self.learned_classes:
                self.learned_classes.append(class_label)
            self.op[class_label] = self.op_sum[class_label] / self.sample_counts[class_label]

        if self.args.save_interim:
            model_dir = os.path.join(self.args.output_dir, "task_models", dataset.NAME, self.args.experiment_id)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(self.net, os.path.join(model_dir, f'task{self.current_task}'))
            torch.save(self.op, os.path.join(model_dir, f'object_ptototypes.ph'))

        print(f"--- Preparing for Task {self.current_task + 1} ---")

        self.current_task += 1

        self.drs_setup_done = False
        self.eval_prototypes = True
        self.flag = True

        for module in self.net.modules():
            if hasattr(module, 'set_task'):
                module.set_task(self.current_task)

        self.get_optimizer()

    def get_optimizer(self):
        if self.current_task == 0:
            print("Task 0: Full network fine-tuning mode.")
            params_to_train = self.net.parameters()

            for param in self.net.parameters():
                param.requires_grad = True

            self.opt = torch.optim.AdamW(params_to_train, lr=1e-4, weight_decay=0.01)
            self.scheduler = None

        else:
            print(f"Task > 0: Parameter-Efficient Continual Learning mode.")
            params_to_train = []
            for name, param in self.net.named_parameters():
                param.requires_grad = False

                if 'head' in name:
                    param.requires_grad = True
                    params_to_train.append(param)

                elif 'lora' in name:
                    parts = name.split('.')
                    if len(parts) > 1 and parts[-2].isdigit():
                        task_idx_in_name = int(parts[-2])
                        if task_idx_in_name == self.current_task:
                            param.requires_grad = True
                            params_to_train.append(param)
            self.opt = torch.optim.AdamW(params_to_train, lr=self.args.lr, weight_decay=0.01)

            if self.args.use_lr_scheduler:
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, self.args.lr_steps, gamma=0.1)
            else:
                self.scheduler = None

        print("Parameters to train:", [name for name, p in self.net.named_parameters() if p.requires_grad])
