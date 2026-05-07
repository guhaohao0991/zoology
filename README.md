# Zoology — 新 Seq Mixer 架构测试与可视化套件

面向 Zoology MQAR 基准，提供一套**增量、可重复**的工作流：新增一个 sequence mixer 架构 → 在三组互补的 MQAR 任务上对比多种基线 → 跨 sweep 合并结果 → 出图 + 结构化摘要。

所有约定沉淀在 `zoology/experiments/mqar_kda/` 下，三条核心流水线：

| 任务 | 数据源 | 考察能力 | config |
|---|---|---|---|
| **MQAR** | `zoology.data.multiquery_ar.MQARConfig` | 基础 in-context recall（2^n kv pairs） | `mqar_configs_random_false.py` |
| **Composition** | `zoology.data.compositional_mqar.CompositionalMQARConfig` | 多键组合（完全平方数 kv pairs） | `composition_configs_random_false.py` |
| **Forgetting** | `zoology.data.forgetting_mqar.ForgettingMQARConfig` | 键覆盖/遗忘能力 | `forgetting_configs.py` |

每组实验在相同 8 架构、3 个 d_model、4 个学习率下扫，共 120 runs。可只开新架构做消融。

---

## 1. Mixer 契约

新 mixer 放在 `zoology/mixers/<new_arch>.py`，必须满足：

- `__init__(d_model, layer_idx=None, **kwargs)` —— 不吃 HF config 对象
- `forward(hidden_states, **kwargs) -> torch.Tensor` —— 返回 bare tensor（不是 tuple）
- 可选 `state_size(**kwargs) -> int`（用于内存报告）

**参考模板**：

- `zoology/mixers/kda.py` —— 纯 KDA，最小完整实现
- `zoology/mixers/fg_gdn.py` —— 多分支（fg_gdn / fg_gdn_plus / fg_gdn_efla / use_xsa_kda）示例，演示如何在同一个类里承载多个变体

注意：若需要像 `inv_dt` 这样的 HF 风格重参数化，显式在 `__init__` 末尾对**该 Parameter 本身**重算即可；不要用 `self.apply(_init_weights)` 全局重初始化 `nn.Linear`，那会覆盖 zoology 的默认 init。

---

## 2. 三步接入新架构

### Step 1 · 实现 mixer

```python
# zoology/mixers/my_mixer.py
import torch
import torch.nn as nn

class MyMixer(nn.Module):
    def __init__(self, d_model, layer_idx=None, num_heads=2, **kwargs):
        super().__init__()
        ...
    def forward(self, hidden_states, **kwargs):
        ...
        return out  # bare tensor
    def state_size(self, sequence_length=2048):
        return ...  # optional
```

**SomkeTest**（前/反向都跑一下，涵盖 `q_len=64` 训练路径——MQAR 的第一个 batch 就是 64）：

```bash
cd /root/paddlejob/workspace/env_run/output/haohao/zoology
conda run -n py310_ljl python -c "
import torch
from zoology.mixers.my_mixer import MyMixer

layer = MyMixer(d_model=128, num_heads=2).cuda().bfloat16()
layer.train()
x = torch.randn(2, 64, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
y = layer(x)
y.float().pow(2).mean().backward()
assert not torch.isnan(y).any()
print('ok', tuple(y.shape))
"
```

### Step 2 · 注册工厂函数

在 `zoology/experiments/models_repo.py` 文末加：

```python
def add_my_mixer(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=2, d_models=None):
    block_type = "TransformerBlock"
    for d_model in (d_models or DEFAULT_D_MODELS):
        my_mixer = dict(
            name="zoology.mixers.my_mixer.MyMixer",
            kwargs={"num_heads": 2, ...},
        )
        mixers = [conv_mixer, my_mixer] if conv_mixer is not None else [my_mixer]
        mixer = ModuleConfig(name="zoology.mixers.hybrid.Hybrid", kwargs={"configs": mixers})
        model = ModelConfig(
            block_type=block_type,
            d_model=d_model,
            n_layers=num_layers,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name="my_mixer",                # 架构名，后面在 included 里引用
            **model_factory_kwargs,
        )
        models.append(model)
    return models
```

**多变体写法**参考 `add_fg_gdn`（循环三组 flags，每组赋不同 `name`）。

### Step 3 · 接入三组实验 config

对 `zoology/experiments/mqar_kda/{mqar,composition,forgetting}_configs*.py` 做两处同步修改：

```python
from zoology.experiments.models_repo import (
    ..., add_my_mixer      # ← 导入
)

models = ...
models = add_my_mixer(models, conv_mixer, input_seq_len, model_factory_kwargs)  # ← 调用

included = [
    ...,
    "my_mixer",            # ← 加入筛选名单
]
```

若只想单独跑新架构做消融，把 `included` 里其它基线名注释掉即可（当前 `fg_gdn` 三变体就是这么做的）。

---

## 3. 运行实验

三组实验互相独立，按需跑：

```bash
cd /root/paddlejob/workspace/env_run/output/haohao/zoology

export WANDB_API_KEY=<your_key>
export https_proxy=<your_proxy>

# ─── 基础 MQAR ─────────────────────────────────────────
conda run -n <conda_env> python -m zoology.launch \
    zoology/experiments/mqar_kda/mqar_configs_random_false.py \
    -p --gpus 0,1,2,3,4,5,6,7

# ─── 组合 MQAR ─────────────────────────────────────────
conda run -n <conda_env> python -m zoology.launch \
    zoology/experiments/mqar_kda/composition_configs_random_false.py \
    -p --gpus 0,1,2,3,4,5,6,7

# ─── Forgetting MQAR ──────────────────────────────────
conda run -n <conda_env> python -m zoology.launch \
    zoology/experiments/mqar_kda/forgetting_configs.py \
    -p --gpus 0,1,2,3,4,5,6,7
```

**参数说明**：

- `--gpus` 是 `CUDA_VISIBLE_DEVICES` 的值（逗号分隔 GPU id，每张 GPU 同时只跑 1 个 run）
- `-p` 启用 Ray 并行
- 启动时 log 头会打印 `sweep_id='<sweep_name>'`，**复制这个完整字符串**，下一步注册要用

120 runs × 3 任务，8 GPU 并行，典型墙钟 1–2 h。

---

## 4. 增量注册 + 可视化

分析脚本 `zoology/experiments/mqar_kda/analyze_results.py` 内置三件套：
**sweep 注册表**（单一真值源）+ **per-sweep 缓存**（只拉一次 wandb）+ **跨 sweep 合并视图**（按 `run_id` 去重）。

### 核心操作：每跑完一个新架构实验，一条命令入库

```bash
conda run -n py310_ljl python zoology/experiments/mqar_kda/analyze_results.py \
    --register MQAR        <mqar_sweep_id> \
    --register Composition <composition_sweep_id> \
    --register Forgetting  <forgetting_sweep_id>
```

执行流程：

1. 把新 sweep_id 追加到 `results/sweep_registry.json`（去重）
2. 仅对新 sweep 访问 wandb，其它 sweep 复用本地缓存 `results/data/cache/<sweep_id>.csv`
3. 按 task concat 所有历史 sweep → 去重 → 写 `results/data/<task>_runs.csv`
4. 重画四张图 + 刷新 `summary.json`

### 其它常用命令

```bash
# 查看当前注册表 + 缓存状态
python .../analyze_results.py --list

# 只用本地缓存重画（不访问 wandb）
python .../analyze_results.py --offline

# 强制重拉某个 sweep（删除缓存）
python .../analyze_results.py --refetch <sweep_id>

# 只分析任务子集
python .../analyze_results.py --tasks MQAR Composition
```

### 手动编辑注册表

直接改 `results/sweep_registry.json`（JSON 结构：`{task: [sweep_id, ...]}`），适合批量增删或回滚。

---

## 5. 输出物

每次 `analyze_results.py` 运行后，结果结构如下：

```
results/
├── sweep_registry.json          # 所有已注册 sweep 的单一真值源
├── data/
│   ├── cache/
│   │   ├── <sweep_id_1>.csv     # 单个 sweep 的 wandb 原始拉取（只拉一次）
│   │   └── <sweep_id_2>.csv
│   ├── mqar_runs.csv            # MQAR 任务合并视图（所有已注册 sweep 的并集）
│   ├── composition_runs.csv
│   ├── forgetting_runs.csv
│   └── eval_matrix.csv          # 长表：Model × d_model × Task × num_kv_pairs × accuracy
├── figures/
│   ├── difficulty_curves.png    # 图A：accuracy vs num_kv_pairs，facet=task，hue=model
│   ├── heatmap.png              # 图B：model × num_kv_pairs 热力图，facet=d_model×task
│   ├── radar.png                # 图C：各模型在最难分片的能力雷达图
│   └── fact_capacity.png        # 图D：acc≥99% 下可解的最大 kv 数 vs d_model
└── summary.json                 # 结构化摘要：arch → d_model → task → {max_kv_solved_99, accuracy_by_kv}
```

**`summary.json` 结构**（便于下游程序消费）：

```json
{
  "meta": {"tasks": ["MQAR", "Composition", "Forgetting"], "threshold": 0.99},
  "models": {
    "fg_gdn": {
      "128": {
        "MQAR":        {"max_kv_solved_99": 64, "accuracy_by_kv": {"4": 1.0, "8": 1.0, ...}},
        "Composition": {...},
        "Forgetting":  {...}
      },
      "64":  {...},
      "32":  {...}
    },
    "kda": {...}
  }
}
```

---

## 6. 典型工作流示例（添加 my_mixer 并对比 kda 基线）

```bash
# 1) 实现 + 冒烟
vim zoology/mixers/my_mixer.py
conda run -n py310_ljl python -c "from zoology.mixers.my_mixer import MyMixer; ..."

# 2) 注册工厂 + 接入 config
vim zoology/experiments/models_repo.py           # 加 add_my_mixer
vim zoology/experiments/mqar_kda/mqar_configs_random_false.py          # import + call + included
vim zoology/experiments/mqar_kda/composition_configs_random_false.py   # 同上
vim zoology/experiments/mqar_kda/forgetting_configs.py                 # 同上

# 3) 启动 3 个实验（留意 log 里的 sweep_id）
conda run -n py310_ljl python -m zoology.launch zoology/experiments/mqar_kda/mqar_configs_random_false.py -p --gpus 0,1,2,3,4,5,6,7
conda run -n py310_ljl python -m zoology.launch zoology/experiments/mqar_kda/composition_configs_random_false.py -p --gpus 0,1,2,3,4,5,6,7
conda run -n py310_ljl python -m zoology.launch zoology/experiments/mqar_kda/forgetting_configs.py -p --gpus 0,1,2,3,4,5,6,7

# 4) 增量入库 + 出图
conda run -n py310_ljl python zoology/experiments/mqar_kda/analyze_results.py \
    --register MQAR <mqar_sweep_id> \
    --register Composition <composition_sweep_id> \
    --register Forgetting <forgetting_sweep_id>

# 5) 查看
open zoology/experiments/mqar_kda/results/figures/heatmap.png
cat zoology/experiments/mqar_kda/results/summary.json
```

---

## 7. 已接入的架构清单

注册工厂: `attention, based, mamba2, delta_net, rwkv7, gla, gated_delta_net, deepseek_nsa, ttt(linear/mlp), kda, fg_gdn(+plus/+efla)`

当前默认 `included` 基线对比组: `attention, based, gla, gated_delta_net, kda` + `fg_gdn` 三变体。取消 `included` 注释即可加回其它基线。

---

## 8. 常见坑

- **Python 环境**：所有 python 命令必须在 `conda run -n py310_ljl` 下，否则 triton / fla 版本不对
- **训练期 mode assert**：若 mixer 里有类似 `if self.training: assert mode == "chunk"` 的检查，auto-switch `if q_len <= 64: mode = "fused_recurrent"` 必须排除训练路径，否则 MQAR 第一个 seq_len=64 的 batch 就会炸（`fg_gdn.py` 已修复，参考其写法）
- **wandb 拉取失败**：需要 `WANDB_API_KEY` + `https_proxy` 环境变量
- **多变体架构**：不要在多个 ModelConfig 里共用同一个 `name`，否则 `best_per_model` 会把它们 groupby 到一起；用不同 `name` 作区分（如 fg_gdn / fg_gdn_plus / fg_gdn_efla）
