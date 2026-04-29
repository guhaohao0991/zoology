# -*- coding: utf-8 -*-
"""
Thin wrappers around flash-linear-attention (FLA) layers for Zoology MQAR evaluation.

Each wrapper satisfies the Zoology mixer contract:
  - __init__(d_model, layer_idx, **kwargs)
  - forward(hidden_states: Tensor[B,T,d]) -> Tensor[B,T,d]   (plain tensor, NOT a tuple)
  - state_size(**kwargs) -> int  (optional, for memory reporting)

FLA layers use `hidden_size` as their primary dim argument; all wrappers
map Zoology's `d_model` to `hidden_size` before delegation.

Supported architectures
-----------------------
  FLAGatedLinearAttention      -- GLA (Gated Linear Attention)
  FLASimpleGatedLinearAttention-- SimpleGLA
  FLADeltaNet                  -- DeltaNet (delta rule)
  FLAGatedDeltaNet             -- GatedDeltaNet
  FLALinearAttention           -- vanilla Linear Attention
  FLAHGRN2                     -- HGRN2
  FLAGatedSlotAttention        -- GSA (Gated Slot Attention)
  FLArwkv6                     -- RWKV-6
  FLAABCAttention              -- ABC (Associative Block Compression)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Generic base: wraps any FLA layer whose forward returns (o, None, cache)
# ---------------------------------------------------------------------------

class _FLALayerWrapper(nn.Module):
    """
    Generic wrapper for FLA layers that return (output, None, past_key_values).
    Subclasses just need to build self.inner in __init__.
    """

    inner: nn.Module

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        out = self.inner(hidden_states)
        # FLA layers return (tensor, None, cache) or bare tensor
        if isinstance(out, tuple):
            return out[0]
        return out

    def state_size(self, **kwargs) -> int:
        if hasattr(self.inner, "state_size"):
            return self.inner.state_size(**kwargs)
        return 0


# ---------------------------------------------------------------------------
# GLA  (fla.layers.GatedLinearAttention)
# ---------------------------------------------------------------------------

class FLAGatedLinearAttention(_FLALayerWrapper):
    """
    Gated Linear Attention (Yang et al., 2023).
    https://arxiv.org/abs/2312.06635
    """

    def __init__(
        self,
        d_model: int = 1024,
        layer_idx: int = None,
        mode: str = "chunk",
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 2,
        use_short_conv: bool = False,
        use_output_gate: bool = True,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        fuse_norm: bool = True,
        **kwargs,
    ):
        super().__init__()
        from fla.layers import GatedLinearAttention
        self.inner = GatedLinearAttention(
            hidden_size=d_model,
            layer_idx=layer_idx,
            mode=mode,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            use_short_conv=use_short_conv,
            use_output_gate=use_output_gate,
            gate_logit_normalizer=gate_logit_normalizer,
            gate_low_rank_dim=gate_low_rank_dim,
            fuse_norm=fuse_norm,
        )


# ---------------------------------------------------------------------------
# SimpleGLA  (fla.layers.simple_gla.SimpleGatedLinearAttention)
# ---------------------------------------------------------------------------

class FLASimpleGatedLinearAttention(_FLALayerWrapper):
    """
    Simplified Gated Linear Attention (no data-dependent gating on keys).
    """

    def __init__(
        self,
        d_model: int = 1024,
        layer_idx: int = None,
        mode: str = "chunk",
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 2,
        use_short_conv: bool = True,
        fuse_norm: bool = True,
        **kwargs,
    ):
        super().__init__()
        from fla.layers.simple_gla import SimpleGatedLinearAttention
        self.inner = SimpleGatedLinearAttention(
            hidden_size=d_model,
            layer_idx=layer_idx,
            mode=mode,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            use_short_conv=use_short_conv,
            fuse_norm=fuse_norm,
        )


# ---------------------------------------------------------------------------
# DeltaNet  (fla.layers.DeltaNet)
# ---------------------------------------------------------------------------

class FLADeltaNet(_FLALayerWrapper):
    """
    DeltaNet: linear attention with the delta rule.
    https://arxiv.org/abs/2406.06484
    """

    def __init__(
        self,
        d_model: int = 1024,
        layer_idx: int = None,
        mode: str = "chunk",
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 2,
        use_beta: bool = True,
        use_gate: bool = False,
        use_short_conv: bool = False,
        qk_norm: str = "l2",
        **kwargs,
    ):
        super().__init__()
        from fla.layers import DeltaNet
        self.inner = DeltaNet(
            hidden_size=d_model,
            layer_idx=layer_idx,
            mode=mode,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            use_beta=use_beta,
            use_gate=use_gate,
            use_short_conv=use_short_conv,
            qk_norm=qk_norm,
        )


# ---------------------------------------------------------------------------
# GatedDeltaNet  (fla.layers.GatedDeltaNet)
# ---------------------------------------------------------------------------

class FLAGatedDeltaNet(_FLALayerWrapper):
    """
    GatedDeltaNet: DeltaNet with an output gate.
    https://arxiv.org/abs/2412.06464
    """

    def __init__(
        self,
        d_model: int = 1024,
        layer_idx: int = None,
        mode: str = "chunk",
        expand_v: float = 2.0,
        num_heads: int = 2,
        use_gate: bool = True,
        use_short_conv: bool = False,
        **kwargs,
    ):
        super().__init__()
        from fla.layers import GatedDeltaNet
        # head_dim derived so that num_heads * head_dim == d_model
        head_dim = max(1, d_model // num_heads)
        self.inner = GatedDeltaNet(
            hidden_size=d_model,
            layer_idx=layer_idx,
            mode=mode,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=num_heads,
            use_gate=use_gate,
            use_short_conv=use_short_conv,
        )


# ---------------------------------------------------------------------------
# LinearAttention  (fla.layers.LinearAttention)
# ---------------------------------------------------------------------------

class FLALinearAttention(_FLALayerWrapper):
    """
    Vanilla linear attention with pluggable feature maps.
    """

    def __init__(
        self,
        d_model: int = 1024,
        layer_idx: int = None,
        mode: str = "chunk",
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 2,
        feature_map: str = "elementwise_product",
        **kwargs,
    ):
        super().__init__()
        from fla.layers import LinearAttention
        self.inner = LinearAttention(
            hidden_size=d_model,
            mode=mode,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            feature_map=feature_map,
        )


# ---------------------------------------------------------------------------
# HGRN2  (fla.layers.HGRN2Attention)
# ---------------------------------------------------------------------------

class FLAHGRN2(_FLALayerWrapper):
    """
    HGRN2: Hierarchical Gated Recurrent Network v2.
    https://arxiv.org/abs/2404.07904

    expand_ratio is the per-head dim (forget_dim = num_heads * expand_ratio).
    We cap it to d_model so num_heads = d_model // expand_ratio >= 1.
    """

    def __init__(
        self,
        d_model: int = 1024,
        layer_idx: int = None,
        mode: str = "chunk",
        expand_ratio: int = 64,  # per-head dim; keep <= d_model
        use_short_conv: bool = False,
        **kwargs,
    ):
        super().__init__()
        from fla.layers import HGRN2Attention
        # ensure at least 1 head
        expand_ratio = min(expand_ratio, d_model)
        self.inner = HGRN2Attention(
            hidden_size=d_model,
            layer_idx=layer_idx,
            mode=mode,
            expand_ratio=expand_ratio,
            use_short_conv=use_short_conv,
        )


# ---------------------------------------------------------------------------
# GSA  (fla.layers.GatedSlotAttention)
# ---------------------------------------------------------------------------

class FLAGatedSlotAttention(_FLALayerWrapper):
    """
    Gated Slot Attention: linear attention with a learned slot memory.
    https://arxiv.org/abs/2409.07146
    """

    def __init__(
        self,
        d_model: int = 1024,
        layer_idx: int = None,
        mode: str = "chunk",
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 2,
        num_slots: Optional[int] = None,
        use_short_conv: bool = False,
        **kwargs,
    ):
        super().__init__()
        from fla.layers import GatedSlotAttention
        self.inner = GatedSlotAttention(
            hidden_size=d_model,
            layer_idx=layer_idx,
            mode=mode,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            num_slots=num_slots or d_model // num_heads,
            use_short_conv=use_short_conv,
        )


# ---------------------------------------------------------------------------
# RWKV-6  (fla.layers.RWKV6Attention)
# ---------------------------------------------------------------------------

class FLARRWKV6(_FLALayerWrapper):
    """
    RWKV-6 linear attention variant.
    https://arxiv.org/abs/2404.05892
    """

    def __init__(
        self,
        d_model: int = 1024,
        layer_idx: int = None,
        mode: str = "chunk",
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 2,
        proj_low_rank_dim: int = 32,
        gate_low_rank_dim: int = 64,
        fuse_norm: bool = True,
        **kwargs,
    ):
        super().__init__()
        from fla.layers import RWKV6Attention
        self.inner = RWKV6Attention(
            hidden_size=d_model,
            layer_idx=layer_idx,
            mode=mode,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            proj_low_rank_dim=proj_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            fuse_norm=fuse_norm,
        )


# ---------------------------------------------------------------------------
# ABC  (fla.layers.ABCAttention)
# ---------------------------------------------------------------------------

class FLAABCAttention(_FLALayerWrapper):
    """
    Associative Block Compression attention.
    https://arxiv.org/abs/2405.02816
    """

    def __init__(
        self,
        d_model: int = 1024,
        layer_idx: int = None,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 2,
        num_slots: Optional[int] = None,
        use_short_conv: bool = False,
        **kwargs,
    ):
        super().__init__()
        from fla.layers import ABCAttention
        self.inner = ABCAttention(
            hidden_size=d_model,
            layer_idx=layer_idx,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            num_slots=num_slots or d_model // num_heads,
            use_short_conv=use_short_conv,
        )


# ---------------------------------------------------------------------------
# KDA  (fla.layers.kda.KimiDeltaAttention)
# ---------------------------------------------------------------------------

class FLAKimiDeltaAttention(_FLALayerWrapper):
    """
    KimiDeltaAttention (KDA): delta-rule attention with Kimi-style gating.
    From the flash-linear-attention library (fla.layers.kda).

    head_dim is derived as d_model // num_heads so that
    num_heads * head_dim == d_model exactly.
    """

    def __init__(
        self,
        d_model: int = 1024,
        layer_idx: int = None,
        mode: str = "chunk",
        expand_v: float = 1.0,
        num_heads: int = 2,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        **kwargs,
    ):
        super().__init__()
        from fla.layers.kda import KimiDeltaAttention
        head_dim = max(1, d_model // num_heads)
        self.inner = KimiDeltaAttention(
            hidden_size=d_model,
            layer_idx=layer_idx,
            mode=mode,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=num_heads,
            use_short_conv=use_short_conv,
            allow_neg_eigval=allow_neg_eigval,
        )


