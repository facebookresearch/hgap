# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from trajectory.utils import watch

#------------------------ base ------------------------#

logbase = '~/logs/'
gpt_expname = 'vae/vq'

## automatically make experiment names for planning
## by labelling folders with these args
args_to_watch = [
    ('prefix', ''),
    ('plan_freq', 'freq'),
    ('horizon', 'H'),
    ('beam_width', 'beam'),
]

base = {
    'train': {
        'type': "prior",
        'logbase': logbase,
        'model': "VQTransformer",
        'tag': "experiment",
        'state_conditional': True,
        #'encoder_inputs': ["state", "action", "reward", "return", "terminal"],
        'encoder_inputs': ["state", "action", "mask"],
        'ae_type': "AttentionCNN",
        'N': 100,
        'discount': 0.995,
        'iql_critic': False,
        'downstream_task': "prior",
        'n_layer': 4,
        'n_head': 4,
        'prior_layer': 4,
        'prior_head': 4,
        'value_layer': 4,
        'value_head': 4,

        'prior_learning_rate': 3e-4,
        "num_workers": 2,
        'latent_step': 4,
        'code_per_step': 8,
        'causal_attention': True,
        'causal_conv': True,
        'n_embd': 128,
        'prior_embd': 128,
        'value_embd': 128,
        'trajectory_embd': 1024,
        'K': 512,
        'blocks_per_layer': 2,
        'load_batch_size': 128,
        'train_batch_size': 128,
        'learning_rate': 2e-4,
        'lr_decay': True,
        'seed': 42,
        'device': 'cuda',
        'n_epochs_ref': 1200,
        'n_saves': 3,
        'tau': 0.8,

        'embd_pdrop': 0.0,
        'resid_pdrop': 0.1,
        'attn_pdrop': 0.1,

        'step': 1,
        'subsampled_sequence_length': 17,
        'termination_penalty': -100,
        'exp_name': gpt_expname,

        'position_weight': 1,
        'action_weight': 5,
        'reward_weight': 0.0,
        'value_weight': 0.0,
        'prior_value_weight': 1.0,

        'suffix': '',

        "normalize": True,
        "normalize_reward": False,
        "max_path_length": 1000,
        "bottleneck": "pooling",
        "masking": "none",
        "disable_goal": False,
        "residual": "absolute",
        "ma_update": True,
        "use_discriminator": False,
        "disc_start": 0.1,

        "position_embedding": "absolute",
        "keep_hdf5s_open": True,
        "n_tokens_target": 1e6,
        # debug
        "debug": False,

        "enable_fp16": True,
        "enable_prior_fp16": True,
        "bootstrap": True,
        "bootstrap_ignore_terminal": False,
        "twohot_value": False,
        "symlog": False,
        "datasource": "auto",
        "ignore_incomplete_episodes": True,

        "validation_episodes": 200,

        "data_parallel": False,
        "pretrained_model": "",

        'relabel_type': "speed",
        'reward_clip': "none",
        'body_height_penalty': 0.0,
        'body_height_limit': 0.8,
        'value_policy_gradient_ratio': 0.5,
        'value_ema_rate': 0.998,
        "prior_gradient_norm_clip": 1.0,
        'initial_value_weight': 0.2,
        "cql_weight": 0.1,

        "prior_name": "", # only used for ciritc training
        "critic_name": "",  # only used for prior finetuning
    },

    'plan': {
        "task": "",
        "vae_name": "",
        "prior_name": "",
        "critic_name": "",
        'discrete': False,
        'logbase': logbase,
        'gpt_loadpath': gpt_expname,
        'gpt_epoch': 'latest',
        'device': 'cuda',
        'renderer': 'Renderer',
        'suffix': '0',

        'plan_freq': 1,
        'horizon': 16,
        'temperature': 1.0,

        "rounds": 2,
        "nb_samples": 4096,
        "k": 16,

        'beam_width': 64,
        'n_expand': 4,

        'prob_threshold': 1e-10,
        'prob_weight': 5e3,

        'advantage_weight': 1.0,
        'top_p': 0.99,
        "discount": 1.0,

        'vis_freq': 200,
        'exp_name': watch(args_to_watch),
        'verbose': True,
        'uniform': False,

        # Planner
        'test_planner': 'beam_prior',
        # debug
        "debug": False,
        # clip name subset of prior training data, e.g. ["CMU_001", "CMU_001"]
        "prior_data_subset": [],
        "objective": 'reward',
    },
}

#------------------------ locomotion ------------------------#

hammer_cloned_v0 = hammer_human_v0 = human_expert_v0 = relocate_cloned_v0 = relocate_human_v0 = relocate_expert_v0 = door_cloned_v0 = door_human_v0 = door_expert_v0 = {
    'train': {
        "termination_penalty": None,
        "max_path_length": 200,
        'n_epochs_ref': 10,
        'subsampled_sequence_length': 25,
        'n_layer': 3,
        'n_embd': 64,
    },
    'plan': {
        'horizon': 24,
    }
}

pen_cloned_v0 = pen_expert_v0 = pen_human_v0 = {
    'train': {
        "termination_penalty": None,
        "max_path_length": 100,
        'n_epochs_ref': 10,
        'subsampled_sequence_length': 25,
        'n_layer': 3,
        'n_embd': 16,
        'K': 32,
        'prior_layer': 2,
        'prior_embd': 64,
        'prior_head': 2,
    },
    'plan': {
        'prob_weight': 5e2,
        'horizon': 24,
    }
}

antmaze_ultra_diverse_v0=antmaze_ultra_play_v0=antmaze_large_diverse_v0=antmaze_large_play_v0=antmaze_medium_diverse_v0=antmaze_medium_play_v0=antmaze_umaze_v0={
    'train':{
        "disable_goal": False,
        "termination_penalty": None,
        "max_path_length": 1001,
        "normalize": False,
        "normalize_reward": False,
        'lr_decay': False,
        'K': 8192,
        "discount": 0.998,
        'value_weight': 0.0001,
        'subsampled_sequence_length': 16,
    },
    'plan': {
        'iql_value': False,
        'horizon': 15,
        'vis_freq': 200,
        'renderer': "AntMazeRenderer",
        'beam_width': 2,
        'n_expand': 4,
}
}

mocapact= {
    'train': {
        'termination_penalty': None,
    }
}
