# -*- coding: utf-8 -*-
"""
KDA smoke test: lightweight MQAR run to validate FLAKimiDeltaAttention + grad clipping.

目的：
  - 验证 KDA 在 Zoology MQAR 框架中不再出现 NaN
  - 验证 gradient clipping (max_norm=1.0) 已生效
  - 使用极小参数快速跑完（<5 min / run）

设置：
  seq_len=128, num_kv_pairs=16, vocab_size=512
  d_model ∈ [128, 256], num_heads=2 (head_dim=64/128)
  lr ∈ [1e-3, 2.2e-3], max_epochs=20
  num_examples_train=10_000, num_examples_test=1_000

对比基线：同配置的 GatedDeltaNet（已知稳定）

Run:
  conda run -n py310_ljl python -m zoology.launch \\
      zoology/experiments/kda_smoketest.py \\
      -p --gpus 4 \\
  2>&1 | tee results/kda_smoketest.log
"""

import uuid
import numpy as np

from zoology.config import (
    TrainConfig, DataConfig, LoggerConfig, ModelConfig, ModuleConfig
)
from zoology.data.multiquery_ar import MQARConfig

sweep_id   = uuid.uuid4().hex[:6]
sweep_name = "fixhd-" + sweep_id
project_name = "fla_test3"

FLA = "zoology.mixers.fla_wrappers"

VOCAB_SIZE  = 2048
SEQ_LEN     = 256
NUM_KV      = 32
D_MODELS    = [32, 64, 128]
H_DIM = 32
LRS         = np.logspace(-4, -3, 4)
GC = 0.5
MAX_EPOCHS  = 40

data = DataConfig(
    train_configs=[MQARConfig(
        num_examples=50_000, vocab_size=VOCAB_SIZE,
        input_seq_len=SEQ_LEN, num_kv_pairs=NUM_KV,
        power_a=0.01, random_non_queries=True,
    )],
    test_configs=[MQARConfig(
        num_examples=2_000, vocab_size=VOCAB_SIZE,
        input_seq_len=SEQ_LEN, num_kv_pairs=NUM_KV,
        power_a=0.01, random_non_queries=False,
    )],
    batch_size=(128, 32),
    cache_dir="/tmp/zoology_kda_smoketest",
)

configs = []

for d_model in D_MODELS:
    num_heads  = d_model // H_DIM # 

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
        )
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
                f"kda_smoke-{arch_name}"
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
    f"[kda_smoketest] sweep_id={sweep_id}  total_runs={len(configs)}\n"
)
