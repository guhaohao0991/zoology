import uuid
import numpy as np
from zoology.config import TrainConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig


sweep_id = uuid.uuid4().hex[:6]
sweep_name = "mqar_configs_random_true_" + sweep_id
PROJ_NAME = "mqar_0513"

VOCAB_SIZE = 8_192
RANDOM_FLAG=True

# 1. First we are going to create the data configuration

train_configs = [    
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=100_000, num_kv_pairs=4, random_non_queries=RANDOM_FLAG),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000, num_kv_pairs=8, random_non_queries=RANDOM_FLAG),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=16, random_non_queries=RANDOM_FLAG),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=32, random_non_queries=RANDOM_FLAG),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=64, random_non_queries=RANDOM_FLAG),
]
test_configs = [
    # MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=4, random_non_queries=RANDOM_FLAG),
    # MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=8, random_non_queries=RANDOM_FLAG),
    # MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=16, random_non_queries=RANDOM_FLAG),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=1_000, num_kv_pairs=32, random_non_queries=RANDOM_FLAG),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=1_000, num_kv_pairs=64, random_non_queries=RANDOM_FLAG),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=1_000, num_kv_pairs=128, random_non_queries=RANDOM_FLAG),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=1_000, num_kv_pairs=256, random_non_queries=RANDOM_FLAG),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=2048, num_examples=1_000, num_kv_pairs=512, random_non_queries=RANDOM_FLAG),
]

input_seq_len=max([c.input_seq_len for c in train_configs + test_configs])
batch_size = 256
data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    # can pass a tuple if you want a different batch size for train and test
    batch_size=(batch_size, batch_size // 8),
    cache_dir="/data/sim/zoology"
)

# 2. Next, we are going to collect all the different model configs we want to sweep
models = []
d_models = [64, 128] #, 256]

model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}), "vocab_size": VOCAB_SIZE,
}

# define this conv outside of if/else block because it is used in multiple models
conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={
        "l_max": input_seq_len,
        "kernel_size": 3,
        "implicit_long_conv": True,
    }
)


from zoology.experiments.models_repo import (
    add_attention, add_sliding_window, add_based, add_mamba2, add_rwkv7,
    add_delta_net, add_gla, add_gated_delta_net, add_deepseek_nsa, add_ttt,
    add_kda_variants, add_mla, add_momentum_delta_net,
    add_adafactor_delta_net,
)

# models = add_attention(models, conv_mixer, input_seq_len, model_factory_kwargs)
# models = add_based(models, conv_mixer, input_seq_len, model_factory_kwargs)
# models = add_mamba2(models, conv_mixer, input_seq_len, model_factory_kwargs)
# models = add_sliding_window(models, conv_mixer, input_seq_len, model_factory_kwargs)
# models = add_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs)
# models = add_rwkv7(models, conv_mixer, input_seq_len, model_factory_kwargs)
# models = add_gla(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_gated_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs, d_models=d_models)
# models = add_deepseek_nsa(models, conv_mixer, input_seq_len, model_factory_kwargs)
# models = add_ttt(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_kda_variants(models, conv_mixer, input_seq_len, model_factory_kwargs, d_models=d_models)
# models = add_mla(models, conv_mixer, input_seq_len, model_factory_kwargs, d_models=d_models)
# models = add_fg_gdn(models, conv_mixer, input_seq_len, model_factory_kwargs, d_models=d_models)
models = add_momentum_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs, d_models=d_models)
models = add_adafactor_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs, d_models=d_models)

# convenience for filtering out
included = [
    # "attention",
    # "sliding_window",
    # "based",
    # "delta_net",
    # "gla",
    # "gated_delta_net",
    # "deepseek_nsa",
    # "ttt_linear", "ttt_mlp"
    # "kda",
    # "fg_gdn",
    # "fg_gdn_plus",
    # "fg_gdn_efla",
    # "ab_conv",
    # "sep_beta",
    # "momentum_delta_net",
    "adafactor_delta_net",
]
# models = [m for m in models if any([i in m.name for i in included])]
models = [m for m in models if m.name in included]

for model in models:
    model.embedding_init_type = "spherical"
    model.learnable_word_embeddings = True


# 3. Finally we'll create a train config for each
configs = []
for model in models:
    for lr in np.logspace(-4, -2, 6):
    # for lr in [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]:
    # for lr in [1e-4, 2e-4, 5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3]:
        run_id = f"{model.name}-d{model.d_model}-lr{lr:.1e}"
        config = TrainConfig(
            model=model,
            data=data,
            learning_rate=lr,
            max_epochs=32,
            logger=LoggerConfig(
                project_name=PROJ_NAME,
                entity="ppsci"
            ),
            slice_keys=["num_kv_pairs"],
            sweep_id=sweep_name,
            run_id=run_id,
            predictions_path=f"/root/paddlejob/workspace/env_run/output/haohao/zoology/predictions/{run_id}",
            collect_predictions=False,
        )
        configs.append(config)


