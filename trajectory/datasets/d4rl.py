# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import gym
import random
from tasks.dmcontrol import CMUHumanoidGymWrapper, DMCWrapper
from tasks.tracking import MocapTrackingGymEnv
from tasks.motion_completion import MotionCompletionGymEnv

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl

def qlearning_dataset_with_timeouts(env, dataset=None, terminate_on_end=False, disable_goal=False, **kwargs):
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    realdone_ = []
    if "infos/goal" in dataset:
        if not disable_goal:
            dataset["observations"] = np.concatenate([dataset["observations"], dataset['infos/goal']], axis=1)
        else:
            dataset["observations"] = np.concatenate([dataset["observations"], np.zeros([dataset["observations"].shape[0], 2], dtype=np.float32)],
                                                     axis=1)
        dataset["rewards"] = dataset["rewards"] - 1

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i+1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        realdone_bool = bool(dataset['terminals'][i])
        if "infos/goal" in dataset:
            final_timestep = True if (dataset['infos/goal'][i] != dataset['infos/goal'][i+1]).any() else False
        else:
            final_timestep = dataset['timeouts'][i]

        if i < N - 1:
            done_bool += final_timestep

        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue  
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        realdone_.append(realdone_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_)[:,None],
        'terminals': np.array(done_)[:,None],
        'realterminals': np.array(realdone_)[:,None],
    }


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def load_environment(name):
    from dm_control.locomotion.tasks import go_to_target, corridors
    from dm_control.locomotion.tasks.reference_pose import tracking
    if name == 'goto':
        CONTROL_TIMESTEP = 0.03
        #constructor = corridors.RunThroughCorridor
        constructor = go_to_target.GoToTarget
        env_ctor = CMUHumanoidGymWrapper.make_env_constructor(constructor)
        task_kwargs = dict(
            physics_timestep=tracking.DEFAULT_PHYSICS_TIMESTEP,
            control_timestep=CONTROL_TIMESTEP,
            moving_target=True,
        )
        environment_kwargs = dict(
            time_limit=CONTROL_TIMESTEP * 1e7,
            random_state=np.random.randint(1e6)
        )
        arena_size = (8., 8.)
        env = env_ctor(
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            arena_size=arena_size
        )
    elif name == 'corridor':
        CONTROL_TIMESTEP = 0.03
        constructor = corridors.RunThroughCorridor
        env_ctor = CMUHumanoidGymWrapper.make_env_constructor(constructor)
        task_kwargs = dict(
            physics_timestep=tracking.DEFAULT_PHYSICS_TIMESTEP,
            control_timestep=CONTROL_TIMESTEP,
            contact_termination=False,
        )
        environment_kwargs = dict(
            time_limit=CONTROL_TIMESTEP * 400,
            random_state=np.random.randint(1e6),
        )
        arena_size = (8., 8.)
        env = env_ctor(
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            arena_size=arena_size
        )
    elif name in ['speed', 'forward', 'rotate_x', 'rotate_y', 'rotate_z', 'x_vel', 'y_vel', 'z_vel', 'negative_z_vel', 'jump',
                  'backward', 'shift_left']:
        from tasks.cmu_relabel import CMURelabelTask
        CONTROL_TIMESTEP = 0.03
        constructor = CMURelabelTask
        env_ctor = CMUHumanoidGymWrapper.make_env_constructor(constructor)
        relabel_type = name
        contact_termination = False if relabel_type in ['rotate_x', 'rotate_z'] else True
        task_kwargs = dict(
            physics_timestep=tracking.DEFAULT_PHYSICS_TIMESTEP,
            control_timestep=CONTROL_TIMESTEP,
            relabel_type=relabel_type,
            contact_termination=contact_termination,
        )

        environment_kwargs = dict(
            time_limit=CONTROL_TIMESTEP * 400,
            random_state=np.random.randint(1e6),
        )
        arena_size = (8., 8.)
        env = env_ctor(
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            arena_size=arena_size
        )
    elif "tracking" in name:
        if ":" in name:
            name, dataset = name.split(":")
        env = MocapTrackingGymEnv(dataset=dataset)
    elif "motioncompletion" in name:
        # set up environment
        ghost_offset = 0.
        prompt_length = 0
        task_kwargs = dict(
            ghost_offset=np.array([ghost_offset, 0., 0.]),
            always_init_at_clip_start=False,
            termination_error_threshold=0.3,
            min_steps=10,
            max_steps=None,
            steps_before_color_change=prompt_length
        )
        if ":" in name:
            name, dataset = name.split(":")
        env = MotionCompletionGymEnv(dataset=dataset, task_kwargs=task_kwargs) 
    elif "." in name:
        domain_name, task_name = name.split(".")
        env = DMCWrapper(domain_name, task_name)
    else:
        with suppress_output():
            wrapped_env = gym.make(name)

        env = wrapped_env.unwrapped
        env.max_episode_steps = wrapped_env._max_episode_steps

    env.name = name
    return env