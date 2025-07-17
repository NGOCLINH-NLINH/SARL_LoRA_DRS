import torch
import torch.nn as nn


class Attention_LoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., lora_r=8,
                 num_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.lora_A_q = nn.ModuleList([nn.Linear(dim, lora_r, bias=False) for _ in range(num_tasks)])
        self.lora_B_q = nn.ModuleList([nn.Linear(lora_r, dim, bias=False) for _ in range(num_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, lora_r, bias=False) for _ in range(num_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(lora_r, dim, bias=False) for _ in range(num_tasks)])

        self.task_id = 0
        self.num_tasks = num_tasks

        for i in range(num_tasks):
            nn.init.zeros_(self.lora_B_q[i].weight)
            nn.init.zeros_(self.lora_B_v[i].weight)

    def set_task(self, task_id):
        if task_id < self.num_tasks:
            self.task_id = task_id

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)

        delta_q = self.lora_B_q[self.task_id](self.lora_A_q[self.task_id](x))
        delta_v = self.lora_B_v[self.task_id](self.lora_A_v[self.task_id](x))

        q, k, v = qkv.chunk(3, dim=-1)

        q = q + delta_q
        v = v + delta_v

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_lora_subtraction_matrix(self, task_idx):
        w_a_q = self.lora_A_q[task_idx].weight
        w_b_q = self.lora_B_q[task_idx].weight
        w_a_v = self.lora_A_v[task_idx].weight
        w_b_v = self.lora_B_v[task_idx].weight

        delta_q = w_b_q @ w_a_q
        delta_v = w_b_v @ w_a_v
        delta_k = torch.zeros_like(delta_q)

        return torch.cat([delta_q, delta_k, delta_v], dim=0)