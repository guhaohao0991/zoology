# -*- coding: utf-8 -*-
# Adapted from fla.layers.kda.KimiDeltaAttention for the Zoology mixer contract.
#
# Deliberate deviations from the HF-style original (documented for audit):
#   - Signature takes (d_model, num_heads, ...) instead of a HF config object.
#   - forward() returns a plain tensor (zoology mixer contract).
#   - Drops attention_mask / unpad / KV-cache paths (zoology trains with
#     equal-len batches and no cache). Core training math is unchanged.
#   - Gate is computed via fused_kda_gate (matches current fla.ops.kda API
#     which no longer accepts A_log/dt_bias inside the kernel).
#   - dt_bias uses the Mamba/GatedDeltaNet-style inverse-softplus
#     reparameterization instead of fla's zero init:
#         dt ~ LogUniform(1e-3, 1e-1);  inv_dt = dt + log(-expm1(-dt))
#     Rationale: fused_kda_gate computes g = -A.exp() * softplus(f_proj(h) + dt_bias),
#     so zero-init gives softplus(0)=log2≈0.69 → aggressive initial decay,
#     whereas inv-softplus init makes softplus(inv_dt)=dt∈(1e-3, 1e-1) →
#     near-identity initial recurrence, which empirically helps retrieval
#     tasks like MQAR retain information early in training. Shape stays
#     (key_dim,) so the kernel contract is unchanged; note this samples
#     dt per-(head, k_dim) slot (finer grain than GDN's per-head dt).
#   - Default expand_v bumped from 1.0 to 2.0 to match zoology's
#     GatedDeltaNet default (wider value head). Override via kwargs.
#   - Variant flags: use_fg_gdn / use_fg_gdn_plus / use_efla / use_xsa_kda /
#     use_sep_beta / use_conv_alpha / use_conv_beta. Migrated from fg_gdn.py;
#     the underlying f/g/A_log/dt_bias shapes follow fla upstream (not
#     fg_gdn's head_dim-bottleneck variant), so numeric reproducibility with
#     old fg_gdn runs is not expected.

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.modules.l2norm import l2norm
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    raise ImportError("Need to install fla: pip install flash-linear-attention")

if TYPE_CHECKING:
    from fla.models.utils import Cache  # noqa: F401


class KimiDeltaAttention(nn.Module):
    """
    Kimi Delta Attention (KDA) — Zoology-native port.

    Mirrors fla.layers.kda.KimiDeltaAttention on the training path.

    Args:
        d_model: Hidden size.
        expand_v: Expansion ratio for the value dimension. Default 1.0.
        num_heads: Number of key/query heads. Default 16.
        num_v_heads: Number of value heads (enables GVA when > num_heads).
            Defaults to num_heads.
        head_dim: Per-head dimension. Defaults to d_model // num_heads.
        mode: 'chunk' (training) or 'fused_recurrent' (short-seq eval).
        use_short_conv: Whether to apply short 1D conv on q/k/v.
        allow_neg_eigval: Multiply beta by 2 to allow negative eigenvalues.
        conv_size: Short convolution kernel size.
        conv_bias: Bias for short convolution.
        layer_idx: Kept for API parity (unused in zoology training).
        norm_eps: Epsilon for FusedRMSNormGated.
    """

    def __init__(
        self,
        d_model: int = 2048,
        expand_v: float = 2.0,
        num_heads: int = 16,
        num_v_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        mode: str = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_fg_gdn: bool = False,
        use_fg_gdn_plus: bool = False,
        use_efla: bool = False,
        use_xsa_kda: bool = False,
        use_sep_beta: bool = False,
        use_conv_alpha: bool = False,
        use_conv_beta: bool = False,
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()

        flag_count = int(use_fg_gdn) + int(use_fg_gdn_plus) + int(use_efla) + int(use_sep_beta)
        assert flag_count <= 1, (
            "At most one of {use_fg_gdn, use_fg_gdn_plus, use_efla, use_sep_beta} may be True, "
            f"got use_fg_gdn={use_fg_gdn}, use_fg_gdn_plus={use_fg_gdn_plus}, "
            f"use_efla={use_efla}, use_sep_beta={use_sep_beta}."
        )

        hidden_size = int(d_model)
        self.hidden_size = hidden_size
        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.expand_v = expand_v

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.use_fg_gdn = use_fg_gdn
        self.use_fg_gdn_plus = use_fg_gdn_plus
        self.use_efla = use_efla
        self.use_xsa_kda = use_xsa_kda
        self.use_sep_beta = use_sep_beta
        self.use_conv_alpha = use_conv_alpha
        self.use_conv_beta = use_conv_beta

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

        if use_conv_alpha:
            self.alpha_conv1d = ShortConvolution(
                hidden_size=hidden_size,
                kernel_size=conv_size,
                activation=None,
            )
        if use_conv_beta:
            self.beta_conv1d = ShortConvolution(
                hidden_size=hidden_size,
                kernel_size=conv_size,
                activation=None,
            )

        # Decay-gate low-rank projection consumed by fused_kda_gate.
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        # b_proj width depends on variant flag:
        #   default / use_efla / use_xsa_kda: per-head scalar beta (fla)
        #   use_fg_gdn:                       per-(head, k_dim)   -> key_dim
        #   use_fg_gdn_plus:                  separate beta_k/beta_v -> key_dim + value_dim
        #   use_sep_beta:                     per-head scalar beta_k + beta_v
        if use_fg_gdn:
            self.b_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        elif use_fg_gdn_plus:
            self.b_proj = nn.Linear(hidden_size, self.key_dim + self.value_dim, bias=False)
        elif use_sep_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads + self.num_v_heads, bias=False)
        else:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16))
        )
        self.A_log._no_weight_decay = True
        dt = torch.exp(
            torch.rand(self.key_dim, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # Output-gate low-rank projection (with bias on final linear).
        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid", eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        _, q_len, _ = hidden_states.shape

        # Training: always self.mode (must be 'chunk'). Eval: short sequences
        # fall back to fused_recurrent.
        if self.training:
            mode = self.mode
            assert mode == "chunk", "Only chunk mode is supported in training."
        else:
            mode = "fused_recurrent" if q_len <= 64 else self.mode

        cu_seqlens = kwargs.get("cu_seqlens", None)

        if self.use_short_conv:
            q, _ = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=None,
                output_final_state=False,
                cu_seqlens=cu_seqlens,
            )
            k, _ = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=None,
                output_final_state=False,
                cu_seqlens=cu_seqlens,
            )
            v, _ = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=None,
                output_final_state=False,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        if self.use_conv_alpha:
            h_for_g, _ = self.alpha_conv1d(
                x=hidden_states,
                cache=None,
                output_final_state=False,
                cu_seqlens=cu_seqlens,
            )
        else:
            h_for_g = hidden_states
        g = self.f_proj(h_for_g)
        g = fused_kda_gate(g, self.A_log, self.head_k_dim, g_bias=self.dt_bias)

        if self.use_conv_beta:
            h_for_beta, _ = self.beta_conv1d(
                x=hidden_states,
                cache=None,
                output_final_state=False,
                cu_seqlens=cu_seqlens,
            )
        else:
            h_for_beta = hidden_states
        beta = self.b_proj(h_for_beta).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        q, k = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        # GVA: broadcast q/k/g up to num_v_heads. Default-branch beta (per-head
        # scalar) is repeated here too; non-default variants carry their own
        # beta shape and are expected to run with num_v_heads == num_heads.
        ratio = 1
        if self.num_v_heads > self.num_heads:
            ratio = self.num_v_heads // self.num_heads
            q, k, g = (repeat(x, "... h d -> ... (h r) d", r=ratio) for x in (q, k, g))

        # Save v_norm for XSA post-hoc subtraction (pre-branch v).
        v_norm = None
        if self.use_xsa_kda:
            v_norm = F.normalize(v.float(), dim=-1).to(v.dtype)

        use_qk_l2norm_in_kernel = True

        if self.use_efla:
            if ratio > 1:
                beta = repeat(beta, "... h -> ... (h r)", r=ratio)
            _lambda = (k * k).sum(dim=-1).clamp(min=1e-6)
            alpha = -torch.expm1(-beta * _lambda) / _lambda
            beta = alpha.to(k.dtype)
            q = l2norm(q)
            k = l2norm(k)
            use_qk_l2norm_in_kernel = False
        elif self.use_fg_gdn:
            # b_proj output: (..., key_dim). Same beta scales both k and v, so
            # requires head_v_dim == head_k_dim (i.e. expand_v == 1.0) and
            # num_v_heads == num_heads.
            q = l2norm(q)
            k = l2norm(k)
            beta_sqrt = beta.sqrt()
            beta_sqrt = rearrange(beta_sqrt, "... (h d) -> ... h d", d=self.head_k_dim)
            k = (k * beta_sqrt).to(k.dtype)
            v = (v * beta_sqrt).to(v.dtype)
            beta = beta.new_ones(k.shape[:3], requires_grad=False)
            use_qk_l2norm_in_kernel = False
        elif self.use_fg_gdn_plus:
            # b_proj output: key_dim + value_dim (separate beta_k / beta_v).
            q = l2norm(q)
            k = l2norm(k)
            beta_k, beta_v = torch.split(beta, [self.key_dim, self.value_dim], dim=-1)
            beta_k = rearrange(beta_k, "... (h d) -> ... h d", d=self.head_k_dim)
            beta_v = rearrange(beta_v, "... (h d) -> ... h d", d=self.head_v_dim)
            k = (k * beta_k.sqrt()).to(k.dtype)
            v = (v * beta_v.sqrt()).to(v.dtype)
            beta = beta_v.new_ones(k.shape[:3], requires_grad=False)
            use_qk_l2norm_in_kernel = False
        elif self.use_sep_beta:
            # b_proj output: num_heads + num_v_heads (per-head scalar beta_k/beta_v).
            q = l2norm(q)
            k = l2norm(k)
            beta_k, beta_v = torch.split(beta, [self.num_heads, self.num_v_heads], dim=-1)
            k = (k * beta_k.sqrt().unsqueeze(-1)).to(k.dtype)
            v = (v * beta_v.sqrt().unsqueeze(-1)).to(v.dtype)
            beta = beta_v.new_ones(k.shape[:3], requires_grad=False)
            use_qk_l2norm_in_kernel = False
        else:
            # Default KDA: beta is per-head scalar; kernel does l2norm internally.
            if ratio > 1:
                beta = repeat(beta, "... h -> ... (h r)", r=ratio)

        if mode == "chunk":
            o, _ = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, _ = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens,
            )

        out_gate = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
        o = self.o_norm(o, out_gate)
        if self.use_xsa_kda:
            o = o - (o * v_norm).sum(dim=-1, keepdim=True) * v_norm
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        return o

    def state_size(self, sequence_length: int = 2048):
        return self.num_v_heads * self.head_k_dim * self.head_v_dim
