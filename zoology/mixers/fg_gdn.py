# -*- coding: utf-8 -*-
# Adapted for Zoology evaluation framework — FG-GDN / KDA variants.
#
# Mirrors the structure of zoology/mixers/kda.py but preserves the original
# fg_gdn configuration switches:
#   * use_fg_gdn        — per-(head,dim) beta on k and v, sqrt-gated
#   * use_fg_gdn_plus   — separate per-(head,dim) beta_k and beta_v
#   * use_efla          — EFLA-style alpha transform on beta using ||k||^2
#   * use_xsa_kda       — orthogonal post-hoc v-projection subtraction
# At most one of {use_fg_gdn, use_fg_gdn_plus, use_efla} may be True.
#
# Deliberate deviations from the HF-style original (documented for audit):
#   - Drops attention_mask / unpad / cache paths (zoology trains with equal-len
#     batches and no KV cache). The core math on the training path is unchanged.
#   - Signature takes (d_model, num_heads, head_dim, ...) instead of a HF config
#     object, to satisfy the Zoology mixer contract.
#   - __init__ only re-initializes A_log and dt_bias (via the original inv_dt
#     reparameterization). nn.Linear weights are left to zoology's model init.

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from einops import rearrange
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


class FgGdnAttention(nn.Module):
    """
    FG-GDN / KDA variants — native Zoology mixer.

    Args:
        d_model: Hidden size.
        num_heads: Number of heads.
        head_dim: Per-head dim. Defaults to d_model // num_heads.
        mode: 'chunk' or 'fused_recurrent'. Same auto-switch as original:
              if q_len <= 64, forward uses fused_recurrent (eval/short-prompt);
              in training only chunk is allowed.
        use_short_conv: Whether to use short convolutions on q/k/v.
        conv_size: Short convolution kernel size.
        conv_bias: Bias for short convolutions.
        use_fg_gdn, use_fg_gdn_plus, use_efla, use_xsa_kda: branch flags
            (see module docstring). At most one of the first three may be True.
        layer_idx: Layer index (kept for API parity; unused in zoology).
        norm_eps: Epsilon for output RMSNorm.
    """

    def __init__(
        self,
        d_model: int = 2048,
        num_heads: int = 16,
        head_dim: Optional[int] = None,
        mode: str = "chunk",
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_fg_gdn: bool = True,
        use_fg_gdn_plus: bool = False,
        use_efla: bool = False,
        use_xsa_kda: bool = False,
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()

        # Mutual exclusion on the three kernel-path flags.
        flag_count = int(use_fg_gdn) + int(use_fg_gdn_plus) + int(use_efla)
        assert flag_count <= 1, (
            "At most one of {use_fg_gdn, use_fg_gdn_plus, use_efla} may be True, "
            f"got use_fg_gdn={use_fg_gdn}, use_fg_gdn_plus={use_fg_gdn_plus}, "
            f"use_efla={use_efla}."
        )
        assert mode in ("chunk", "fused_recurrent"), f"Not supported mode `{mode}`."

        hidden_size = int(d_model)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        # fg_gdn keeps key and value head dims equal
        self.head_k_dim = self.head_dim
        self.num_k_heads = num_heads

        self.mode = mode
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.layer_idx = layer_idx
        self.norm_eps = norm_eps

        self.use_fg_gdn = use_fg_gdn
        self.use_fg_gdn_plus = use_fg_gdn_plus
        self.use_efla = use_efla
        self.use_xsa_kda = use_xsa_kda

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads
        self.projection_k_size = projection_k_size
        self.projection_size = projection_size

        # q/k/v projections + short convs
        self.q_proj = nn.Linear(hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, projection_size, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=projection_k_size, kernel_size=conv_size,
                bias=conv_bias, activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=projection_k_size, kernel_size=conv_size,
                bias=conv_bias, activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=projection_size, kernel_size=conv_size,
                bias=conv_bias, activation="silu",
            )

        # Decay-gate low-rank projection (f_a -> f_b), consumed by fused_kda_gate.
        self.f_a_proj = nn.Linear(hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        # A_log / dt_bias — shapes match fused_kda_gate expectations.
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16))
            .view(1, 1, -1, 1)
        )
        self.A_log._no_weight_decay = True
        self.dt_bias = nn.Parameter(
            torch.empty(projection_size, dtype=torch.float32).fill_(-1.5)
        )
        self.dt_bias._no_weight_decay = True

        # Beta projection — width depends on branch.
        if use_fg_gdn:
            self.b_proj = nn.Linear(hidden_size, projection_k_size, bias=False)
        elif use_fg_gdn_plus:
            self.b_proj = nn.Linear(hidden_size, projection_k_size + projection_size, bias=False)
        else:
            # default KDA or use_efla
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # Output-gate low-rank projection (g_a -> g_b) feeds FusedRMSNormGated.
        self.g_a_proj = nn.Linear(hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=norm_eps, activation="sigmoid"
        )
        self.o_proj = nn.Linear(projection_size, hidden_size, bias=False)

        # Re-parameterize dt_bias via the original inv_dt trick (critical for
        # fg_gdn to behave as designed). Leave nn.Linear weights to zoology's
        # model-level init.
        self._reparam_special_params()

    def _reparam_special_params(self):
        with torch.no_grad():
            # A_log: re-draw log(Uniform(1,16)), matches original _init_weights.
            self.A_log.copy_(
                nn.init.uniform_(self.A_log, a=1, b=16).log()
            )
            # dt_bias: dt = exp(U * (log0.1 - log0.001) + log0.001).clamp(1e-4)
            #          inv_dt = dt + log(-expm1(-dt))
            dt = torch.exp(
                nn.init.uniform_(self.dt_bias.clone())
                * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            ).clamp(min=1e-4)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias.copy_(inv_dt)

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        _, q_len, _ = hidden_states.shape

        # Same auto-switch as the HF original: short sequences prefer
        # fused_recurrent; in training chunk is enforced.
        mode = "fused_recurrent" if q_len <= 64 else self.mode
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        cu_seqlens = kwargs.get("cu_seqlens", None)

        # --- q/k/v + short conv -----------------------------------------
        if self.use_short_conv:
            q, _ = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=None, output_final_state=False, cu_seqlens=cu_seqlens,
            )
            k, _ = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=None, output_final_state=False, cu_seqlens=cu_seqlens,
            )
            v, _ = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=None, output_final_state=False, cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        # --- decay gate g (log-space), pre-gated via fused_kda_gate ------
        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = fused_kda_gate(g, self.A_log, self.head_dim, g_bias=self.dt_bias)

        beta = self.b_proj(hidden_states).float().sigmoid()

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        # Save v_norm for XSA post-hoc subtraction.
        v_norm = None
        if self.use_xsa_kda:
            v_norm = F.normalize(v.float(), dim=-1).to(v.dtype)

        use_qk_l2norm_in_kernel = True

        # --- branch-specific beta / k / v transforms --------------------
        if self.use_efla:
            # beta: [B,L,H]; EFLA alpha = -expm1(-beta * ||k||^2) / ||k||^2
            _lambda = (k * k).sum(dim=-1, keepdim=False).clamp(min=1e-6)
            alpha = -torch.expm1(-beta * _lambda) / _lambda
            beta = alpha.to(k.dtype)
            q = l2norm(q)
            k = l2norm(k)
            use_qk_l2norm_in_kernel = False
        elif self.use_fg_gdn:
            # beta: [B,L,H*Dk]
            q = l2norm(q)
            k = l2norm(k)
            beta_sqrt = beta.sqrt()
            beta_sqrt = rearrange(beta_sqrt, "... (h d) -> ... h d", d=self.head_k_dim)
            k = (k * beta_sqrt).to(k.dtype)
            v = (v * beta_sqrt).to(v.dtype)
            beta = beta.new_ones(k.shape[:3], requires_grad=False)
            use_qk_l2norm_in_kernel = False
        elif self.use_fg_gdn_plus:
            # beta: [B,L,H*Dk + H*Dv]
            q = l2norm(q)
            k = l2norm(k)
            beta_k, beta_v = torch.split(
                beta, [self.projection_k_size, self.projection_size], dim=-1
            )
            beta_k = rearrange(beta_k, "... (h d) -> ... h d", d=self.head_k_dim)
            beta_v = rearrange(beta_v, "... (h d) -> ... h d", d=self.head_dim)
            k = (k * beta_k.sqrt()).to(k.dtype)
            v = (v * beta_v.sqrt()).to(v.dtype)
            beta = beta.new_ones(k.shape[:3], requires_grad=False)
            use_qk_l2norm_in_kernel = False
        # else: default KDA branch, beta: [B,L,H], use_qk_l2norm_in_kernel=True

        # --- kernel ------------------------------------------------------
        if mode == "chunk":
            o, _ = chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, _ = fused_recurrent_kda(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens,
            )

        # --- output gate + RMSNorm --------------------------------------
        out_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        out_gate = rearrange(out_gate, "... (h d) -> ... h d", d=self.head_dim)
        o = self.o_norm(o, out_gate)

        # --- XSA post-hoc subtraction (orthogonal) ----------------------
        if self.use_xsa_kda:
            o = o - (o * v_norm).sum(dim=-1, keepdim=True) * v_norm

        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        return o

    def state_size(self, sequence_length: int = 2048):
        return self.num_heads * self.head_k_dim * self.head_dim
