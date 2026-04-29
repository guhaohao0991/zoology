# -*- coding: utf-8 -*-
"""
Compositional MQAR smoke test: 测试模型的组合推理能力。

目的：
  - 模型必须同时关注 (K1, K2) 复合键才能检索正确 value
  - 单独看任一 key 都无法确定答案（grid 结构保证无捷径）
  - 多难度扫描：num_kv_pairs ∈ [4, 9, 16, 25]（2²→5² grid）

设置：
  seq_len=256, vocab_size=2048
  d_model ∈ [32, 64, 128], head_dim=32
  lr ∈ logspace(-4, -3, 4), max_epochs=40, grad_clip=0.5

对比模型：KDA, GatedDeltaNet, GLA, GSA, MHA

Run:
  conda run -n py310_ljl python -m zoology.launch \
      zoology/experiments/fla_fixhead/composition_config.py \
      -p --gpus 4
"""

import uuid
import numpy as np

from zoology.config import (
    TrainConfig, DataConfig, LoggerConfig, ModelConfig, ModuleConfig
)
from zoology.data.compositional_mqar import CompositionalMQARConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "fixhd-comp-" + sweep_id
project_name = "fla_composition"

FLA = "zoology.mixers.fla_wrappers"

VOCAB_SIZE = 2048
SEQ_LEN = 256
D_MODELS = [32, 64, 128]
H_DIM = 32
LRS = np.logspace(-4, -3, 4)
GC = 0.5
MAX_EPOCHS = 40

# --- 多难度数据配置 ---
# num_kv_pairs 必须为完全平方数: 4(2x2), 9(3x3), 16(4x4), 25(5x5)
# 约束: num_kv_pairs * 6 <= SEQ_LEN → max = 42, 25*6=150 ✅
KV_PAIRS_SWEEP = [4, 9, 16, 25]

train_configs = [
    CompositionalMQARConfig(
        vocab_size=VOCAB_SIZE, input_seq_len=SEQ_LEN,
        num_examples=20_000, num_kv_pairs=kv,
        power_a=0.01, random_non_queries=False,
    )
    for kv in KV_PAIRS_SWEEP
]

test_configs = [
    CompositionalMQARConfig(
        vocab_size=VOCAB_SIZE, input_seq_len=SEQ_LEN,
        num_examples=1_000, num_kv_pairs=kv,
        power_a=0.01, random_non_queries=False,
    )
    for kv in KV_PAIRS_SWEEP
]

data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    batch_size=(128, 32),
    cache_dir="/tmp/zoology_composition",
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
                f"comp-{arch_name}"
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
    f"[composition_config] sweep_id={sweep_id}  total_runs={len(configs)}\n"
)
