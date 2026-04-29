# -*- coding: utf-8 -*-
"""
Forgetting MQAR smoke test: 测试模型的遗忘/覆盖能力。

目的：
  - 同一 key 先后被赋予不同 value，模型必须记住最后一次的值
  - 测试模型能否正确覆盖旧关联、保留未更新的关联
  - 多难度扫描：num_kv_pairs ∈ [4, 8, 16, 32]，num_updates = kv//2

设置：
  seq_len=256, vocab_size=2048
  d_model ∈ [32, 64, 128], head_dim=32
  lr ∈ logspace(-4, -3, 4), max_epochs=40, grad_clip=0.5

对比模型：KDA, GatedDeltaNet, GLA, GSA, MHA

Run:
  conda run -n py310_ljl python -m zoology.launch \
      zoology/experiments/fla_fixhead/forgetting_config.py \
      -p --gpus 4
"""

import uuid
import numpy as np

from zoology.config import (
    TrainConfig, DataConfig, LoggerConfig, ModelConfig, ModuleConfig
)
from zoology.data.forgetting_mqar import ForgettingMQARConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "fixhd-forget-" + sweep_id
project_name = "fla_forgetting"

FLA = "zoology.mixers.fla_wrappers"

VOCAB_SIZE = 2048
SEQ_LEN = 256
D_MODELS = [32, 64, 128]
H_DIM = 32
LRS = np.logspace(-4, -3, 4)
GC = 0.5
MAX_EPOCHS = 40

# --- 多难度数据配置 ---
# (num_kv_pairs, num_updates): 50% key 被覆盖
# 约束: (kv + updates)*2 + kv*2 <= SEQ_LEN
#   kv=32, updates=16 → (32+16)*2 + 32*2 = 96+64 = 160 <= 256 ✅
KV_SWEEP = [
    (4, 2),
    (8, 4),
    (16, 8),
    (32, 16),
]

train_configs = [
    ForgettingMQARConfig(
        vocab_size=VOCAB_SIZE, input_seq_len=SEQ_LEN,
        num_examples=20_000, num_kv_pairs=kv, num_updates=upd,
        power_a=0.01, random_non_queries=False,
    )
    for kv, upd in KV_SWEEP
]

test_configs = [
    ForgettingMQARConfig(
        vocab_size=VOCAB_SIZE, input_seq_len=SEQ_LEN,
        num_examples=1_000, num_kv_pairs=kv, num_updates=upd,
        power_a=0.01, random_non_queries=False,
    )
    for kv, upd in KV_SWEEP
]

data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    batch_size=(128, 32),
    cache_dir="/tmp/zoology_forgetting",
)

# --- 模型配置 ---
configs = []

for d_model in D_MODELS:
    num_heads = d_model // H_DIM

    archs = {
        "kda": (
            f"{FLA}.FLAKimiDeltaAttention",
            {"num_heads": num_heads},
            "TransformerBlock",
        ),
        "gated_delta_net": (
            f"{FLA}.FLAGatedDeltaNet",
            {"num_heads": num_heads},
            "TransformerBlock",
        ),
        "gla": (
            f"{FLA}.FLAGatedLinearAttention",
            {"num_heads": num_heads},
            "TransformerBlock",
        ),
        "gsa": (
            f"{FLA}.FLAGatedSlotAttention",
            {"num_heads": num_heads},
            "TransformerBlock",
        ),
        "mha": (
            "zoology.mixers.attention.MHA",
            {"num_heads": num_heads},
            "TransformerBlock",
        ),
    }

    for arch_name, (mixer_path, mixer_kwargs, block_type) in archs.items():
        model = ModelConfig(
            block_type=block_type,
            d_model=d_model,
            n_layers=2,
            sequence_mixer=ModuleConfig(name=mixer_path, kwargs=mixer_kwargs),
            state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
            max_position_embeddings=0,
            vocab_size=VOCAB_SIZE,
            name=arch_name,
        )
        for lr in LRS:
            run_id = (
                f"forget-{arch_name}"
                f"-d{d_model}-h{num_heads}"
                f"-lr{lr:.1e}"
                f"-{sweep_id}"
            )
            configs.append(TrainConfig(
                model=model,
                data=data,
                learning_rate=lr,
                grad_norm_clip=GC,
                max_epochs=MAX_EPOCHS,
                seed=42,
                logger=LoggerConfig(
                    project_name=project_name,
                    entity="ppsci",
                ),
                slice_keys=["num_kv_pairs"],
                sweep_id=sweep_name,
                run_id=run_id,
            ))

print(
    f"[forgetting_config] sweep_id={sweep_id}  total_runs={len(configs)}\n"
)
