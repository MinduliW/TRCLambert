"""TRC-style recurrent actor-critic for Lambert RL."""

import torch
import torch.nn as nn

from trc_core import NetConfig, ReasoningModule, make_mlp


class TRCRecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, net: NetConfig):
        super().__init__()
        self.net = net
        d_z, d_h = net.d_z, net.d_h

        self.state_encoder = make_mlp(obs_dim, d_h, d_z)
        self.ctrl_embed = nn.Linear(act_dim, d_z)
        self.reason = ReasoningModule(d_z, d_h, net.n_heads, net.n_blocks, net.dropout)

        self.h_proj = nn.Linear(d_z, d_z)
        self.l_proj = nn.Linear(d_z, d_z)
        self.H_init = nn.Parameter(torch.randn(d_z) * 0.02)
        self.L_init = nn.Parameter(torch.randn(d_z) * 0.02)

        self.mu_head = make_mlp(d_z + act_dim, d_h, act_dim)
        self.v_head = make_mlp(d_z, d_h, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def init_memory(self, obs: torch.Tensor):
        z0 = self.state_encoder(obs)
        z_h = self.H_init.expand(obs.shape[0], -1) + self.h_proj(z0)
        z_l = self.L_init.expand(obs.shape[0], -1) + self.l_proj(z0)
        return z_h, z_l

    def forward_step(self, obs: torch.Tensor, z_h: torch.Tensor, z_l: torch.Tensor, dv_curr: torch.Tensor):
        z0 = self.state_encoder(obs)
        z_ctrl = self.ctrl_embed(dv_curr)
        z_l_next = self.reason(z_l, z_h, z0, z_ctrl)
        z_h_next = self.reason(z_h, z_l_next, z0)

        mu = self.mu_head(torch.cat([z_h_next, dv_curr], dim=-1))
        v = self.v_head(z_h_next).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std, v, z_h_next, z_l_next
