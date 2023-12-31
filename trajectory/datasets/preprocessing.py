# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from trajectory.datasets.mocapact import CMU_HUMANOID_OBSERVABLES

def kitchen_preprocess_fn(observations):
    ## keep first 30 dimensions of 60-dimension observations
    keep = observations[:, :30]
    remove = observations[:, 30:]
    assert (remove.max(0) == remove.min(0)).all(), 'removing important state information'
    return keep

def ant_preprocess_fn(observations):
    qpos_dim = 13 ## root_x and root_y removed
    qvel_dim = 14
    cfrc_dim = 84
    assert observations.shape[1] == qpos_dim + qvel_dim + cfrc_dim
    keep = observations[:, :qpos_dim + qvel_dim]
    return keep

def vmap(fn):

    def _fn(inputs):
        if isinstance(inputs, dict):
            return_1d = False
        else:
            if inputs.ndim == 1:
                inputs = inputs[None]
                return_1d = True
            else:
                return_1d = False

        outputs = fn(inputs)

        if return_1d:
            return outputs.squeeze(0)
        else:
            return outputs

    return _fn

def preprocess_dataset(preprocess_fn):

    def _fn(dataset):
        for key in ['observations', 'next_observations']:
            dataset[key] = preprocess_fn(dataset[key])
        return dataset

    return _fn

def humanoid_preprocess_fn(obs_dict):
    obs_list = []
    for key in CMU_HUMANOID_OBSERVABLES:
        obs_list.append(obs_dict[key])
    obs = np.concatenate(obs_list)
    return obs

def dmcontrol_preprocess_fn(obs_dict):
    obs_list = []
    for key in obs_dict.keys():
        obs_list.append(obs_dict[key])
    obs = np.concatenate(obs_list)
    return obs

preprocess_functions = {
    'kitchen-complete-v0': vmap(kitchen_preprocess_fn),
    'kitchen-mixed-v0': vmap(kitchen_preprocess_fn),
    'kitchen-partial-v0': vmap(kitchen_preprocess_fn),
    'ant-expert-v2': vmap(ant_preprocess_fn),
    'ant-medium-expert-v2': vmap(ant_preprocess_fn),
    'ant-medium-replay-v2': vmap(ant_preprocess_fn),
    'ant-medium-v2': vmap(ant_preprocess_fn),
    'ant-random-v2': vmap(ant_preprocess_fn),
    'speed': vmap(humanoid_preprocess_fn),
    'forward': vmap(humanoid_preprocess_fn),
    'rotate_x': vmap(humanoid_preprocess_fn),
    'rotate_y': vmap(humanoid_preprocess_fn),
    'rotate_z': vmap(humanoid_preprocess_fn),
    'x_vel': vmap(humanoid_preprocess_fn),
    'y_vel': vmap(humanoid_preprocess_fn),
    'jump': vmap(humanoid_preprocess_fn),
    'shift_left': vmap(humanoid_preprocess_fn),
    'backward': vmap(humanoid_preprocess_fn),
    'z_vel': vmap(humanoid_preprocess_fn),
    'negative_z_vel': vmap(humanoid_preprocess_fn),
    'tracking': vmap(humanoid_preprocess_fn),
}

dataset_preprocess_functions = {
    k: preprocess_dataset(fn) for k, fn in preprocess_functions.items()
}

def get_preprocess_fn(env):
    return preprocess_functions.get(env, lambda x: x)