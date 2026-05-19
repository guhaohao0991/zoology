# -*- coding: utf-8 -*-
# Adapted from fla.layers.adafactor_deltanet.AdafactorDeltaNet
# (/root/paddlejob/gpfsspace/hesensen/flash-linear-attention/fla/layers/adafactor_deltanet.py)
# for the Zoology mixer contract.
#
# Deliberate deviations from the upstream fla module:
#   - Signature takes (d_model, num_heads, ...) instead of hidden_size kwarg.
#   - forward() returns a plain tensor (zoology mixer contract).
#   - Drops attention_mask / unpad / KV-cache paths (zoology trains with
#     equal-len batches and no cache). Core training math is unchanged.
#   - Default head_dim derives from d_model // num_heads (zoology-style)
#     instead of the upstream 256, since zoology d_models are 32~256.
#   - Removes the upstream debug print statements in forward().
#   - Adds state_size() for zoology plotting.

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

try:
    from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
    from fla.ops.adafactor_delta_rule import (
        chunk_adafactor_rule,
        fused_recurrent_adafactor_rule,
    )
    from fla.ops.adafactor_delta_rule.naive import chunk_adafactor_delta_rule_ref
except ImportError:
    raise ImportError("Need to install fla: pip install flash-linear-attention")


class AdafactorDeltaNet(nn.Module):
    """Adafactor-style Delta Net — Zoology-native port.

    Mirrors fla.layers.adafactor_deltanet.AdafactorDeltaNet on the training
    path. Uses the naive `chunk_adafactor_delta_rule_ref` reference kernel
    in chunk mode (matching the upstream file), and the fused recurrent
    kernel for short-seq eval.
    """

    def __init__(
        self,
        d_model: int = 256,
        expand_v: float = 2.0,
        num_heads: int = 2,
        num_v_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        mode: str = "chunk",
        use_gate: bool = True,
        use_lora_gate: bool = False,
        lora_gate_rank: Optional[int] = None,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        min_log_mu: float = -2.0,
        tau_factor: int = 1,
        use_output_correction: bool = True,
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()

        hidden_size = int(d_model)
        self.hidden_size = hidden_size
        self.mode = mode
        self.expand_v = expand_v
        self.min_log_mu = min_log_mu

        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.use_output_correction = use_output_correction

        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // self.num_heads

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value_dim "
                f"(got {self.num_v_heads * self.head_dim * expand_v}, expected {self.value_dim})."
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}."
            )
        if not math.isclose(self.head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer head_v_dim "
                f"(got {self.head_dim * expand_v}, expected {self.head_v_dim})."
            )
        assert mode in ("chunk", "fused_recurrent"), f"Not supported mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Adafactor-style decay/momentum projections (per v-head scalars).
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.m_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.e_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        self.a_min_init = 1.0
        self.a_max_init = 16.0
        self.m_min_init = 1.0
        self.m_max_init = 16.0
        self.factor_scale = 4.0
        self.tau = math.sqrt(self.hidden_size / tau_factor)

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(self.a_min_init, self.a_max_init)
        self.A_log = nn.Parameter(torch.log(A.clone()))
        self.A_log._no_weight_decay = True

        dt_min = 0.01
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt.clone())
        self.mu_bias = nn.Parameter(inv_dt.clone())
        self.dt_bias._no_weight_decay = True
        self.mu_bias._no_weight_decay = True

        Mu = torch.empty(self.num_heads, dtype=torch.float32).uniform_(self.m_min_init, self.m_max_init)
        self.Mu_log = nn.Parameter(torch.log(Mu.clone()))
        self.Mu_log._no_weight_decay = True

        self.log_factor = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, self.factor_scale))
        )
        self.log_factor._no_weight_decay = True

        if use_output_correction:
            self.D_log = nn.Parameter(
                torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 1))
            )
            self.D_log._no_weight_decay = True

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off unless you know what you are doing."
            )

        if use_gate:
            if use_lora_gate:
                rank = self.head_v_dim if lora_gate_rank is None else lora_gate_rank
                self.g_proj = nn.Sequential(
                    nn.Linear(hidden_size, rank, bias=False),
                    nn.Linear(rank, self.value_dim, bias=True),
                )
            else:
                self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid", eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        _, q_len, _ = hidden_states.shape

        if self.training:
            mode = self.mode
            assert mode == "chunk", "Only chunk mode is supported in training."
        else:
            mode = "fused_recurrent" if q_len <= 64 else self.mode

        cu_seqlens = kwargs.get("cu_seqlens", None)

        if self.use_short_conv:
            q, _ = self.q_conv1d(
                x=self.q_proj(hidden_states), cache=None, output_final_state=False, cu_seqlens=cu_seqlens,
            )
            k, _ = self.k_conv1d(
                x=self.k_proj(hidden_states), cache=None, output_final_state=False, cu_seqlens=cu_seqlens,
            )
            v, _ = self.v_conv1d(
                x=self.v_proj(hidden_states), cache=None, output_final_state=False, cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        q, k = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            ratio = self.num_v_heads // self.num_heads
            q, k = (repeat(x, "... h d -> ... (h r) d", r=ratio) for x in (q, k))

        a = self.a_proj(hidden_states).float()
        b = self.b_proj(hidden_states).float()
        m = self.m_proj(hidden_states).float()
        e = self.e_proj(hidden_states).float() / self.tau

        log_mu = -self.Mu_log.float().exp() * F.softplus(m + self.mu_bias)
        log_alpha = -self.A_log.float().exp() * F.softplus(a + self.dt_bias)
        eta = F.tanh(e) + 1
        beta = F.sigmoid(b)

        # Angular constraint shared with MomentumDeltaNet.
        theta = torch.arctan(eta * self.log_factor.float().exp())
        beta_upper = torch.sin(theta) ** 2
        alpha_upper = torch.cos(theta) ** 2
        beta = beta_upper * beta
        log_alpha = alpha_upper.log() + log_alpha

        # Hard upper bound on recurrent time-constants (matches upstream).
        LOG_RECURR_MAX = math.log(0.95)
        log_alpha = log_alpha.clamp(max=LOG_RECURR_MAX)
        log_mu = log_mu.clamp(max=LOG_RECURR_MAX)

        if self.min_log_mu is not None:
            log_mu = log_mu.clamp_min(self.min_log_mu)

        if self.use_output_correction:
            q = (q - self.D_log.float().exp()[None, None, :, None] * k).to(k.dtype)

        if mode == "chunk":
            # Manual l2norm: the naive ref kernel does not l2norm internally.
            q_n = q / (q.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6)
            k_n = k / (k.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6)
            o, _ = chunk_adafactor_delta_rule_ref(
                q=q_n, k=k_n, v=v,
                log_alpha=log_alpha, log_mu=log_mu,
                beta=beta, eta=eta,
                initial_S=None, initial_R=None, initial_C=None,
                output_final_state=False,
                chunk_size=64,
            )
        elif mode == "fused_recurrent":
            o, _ = fused_recurrent_adafactor_rule(
                q=q, k=k, v=v,
                log_alpha=log_alpha, log_mu=log_mu,
                beta=beta, eta=eta,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)

        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o.to(self.o_proj.weight.dtype))
        return o

    def state_size(self, sequence_length: int = 2048):
        return self.num_v_heads * self.head_k_dim * self.head_v_dim
