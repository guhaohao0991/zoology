# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

try:
    from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
    from fla.ops.momentum_delta_rule import chunk_mode_rule, fused_recurrent_mode_rule
except ImportError:
    raise ImportError("Need to install fla: pip install flash-linear-attention")

if TYPE_CHECKING:
    from fla.models.utils import Cache


class MomentumDeltaNet(nn.Module):

    def __init__(
        self,
        d_model: int = 256,
        expand_v: float = 2,
        num_heads: int = 2,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        use_p_times_alpha: bool = True,
        use_output_correction: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        min_log_mu: float = -2.,
        tau_factor: int = 1,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs
    ) -> MomentumDeltaNet:
        super().__init__()

        self.mode = mode
        hidden_size = int(d_model)
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.min_log_mu = min_log_mu

        self.use_gate = use_gate
        self.use_p_times_alpha = use_p_times_alpha
        self.use_short_conv = use_short_conv
        self.use_output_correction = use_output_correction
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        head_dim = self.head_dim

        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_heads * self.head_v_dim
        self.layer_idx = layer_idx

        self.a_min_init = 0
        self.a_max_init = 16
        self.m_min_init = 0
        self.m_max_init = 16
        self.factor_scale = 4
        self.tau = math.sqrt(self.hidden_size / tau_factor)

        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.m_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.e_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(self.a_min_init, self.a_max_init)
        self.A_log = nn.Parameter(torch.log(A.clone()))
        self.A_log._no_weight_decay = True

        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt.clone())
        self.mu_bias = nn.Parameter(inv_dt.clone())
        self.dt_bias._no_weight_decay = True
        self.mu_bias._no_weight_decay = True

        Mu = torch.empty(self.num_heads, dtype=torch.float32).uniform_(self.m_min_init, self.m_max_init)
        self.Mu_log = nn.Parameter(torch.log(Mu.clone()))
        self.Mu_log._no_weight_decay = True

        self.log_factor = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, self.factor_scale)))
        self.log_factor._no_weight_decay = True

        if use_output_correction:
            self.D_log = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 1)))
            self.D_log._no_weight_decay = True

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off unless you know what you are doing."
            )

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, activation='sigmoid', eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> torch.Tensor:
        mode = self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            q, conv_state_q = self.q_conv1d(x=self.q_proj(hidden_states), mask=conv_mask, cache=conv_state_q, output_final_state=use_cache)
            k, conv_state_k = self.k_conv1d(x=self.k_proj(hidden_states), mask=conv_mask, cache=conv_state_k, output_final_state=use_cache)
            v, conv_state_v = self.v_conv1d(x=self.v_proj(hidden_states), mask=conv_mask, cache=conv_state_v, output_final_state=use_cache)
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        q, k = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

        a = self.a_proj(hidden_states).float()
        b = self.b_proj(hidden_states).float()
        m = self.m_proj(hidden_states).float()
        e = self.e_proj(hidden_states).float() / self.tau

        log_mu = -self.Mu_log.float().exp() * F.softplus(m + self.mu_bias)
        log_alpha = -self.A_log.float().exp() * F.softplus(a + self.dt_bias)
        eta = F.tanh(e) + 1
        beta = F.sigmoid(b)

        theta = torch.arctan(eta * self.log_factor.float().exp())
        beta_upper = torch.sin(theta) ** 2
        alpha_upper = torch.cos(theta) ** 2
        beta = beta_upper * beta
        log_alpha = alpha_upper.log() + log_alpha

        if self.min_log_mu is not None:
            log_mu.clamp_min_(self.min_log_mu)

        if self.use_output_correction:
            q = (q - self.D_log.float().exp()[None, None, :, None] * k).to(k.dtype)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        cu_seqlens = kwargs.get('cu_seqlens', None)

        if mode == 'chunk':
            o, recurrent_state = chunk_mode_rule(
                q=q, k=k, v=v,
                log_alpha=log_alpha,
                beta=beta,
                log_mu=log_mu,
                eta=eta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                use_p_times_alpha=self.use_p_times_alpha
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_mode_rule(
                q=q, k=k, v=v,
                log_alpha=log_alpha,
                beta=beta,
                log_mu=log_mu,
                eta=eta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                use_p_times_alpha=self.use_p_times_alpha
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o

    def state_size(self, sequence_length: int = 2048):
        return self.num_heads * self.head_k_dim * self.head_v_dim
