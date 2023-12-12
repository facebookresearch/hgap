# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Wraps the MultiClipMocapTracking dm_env into a Gym environment.
"""
import numpy as np
from gym import spaces
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.tasks.reference_pose import types
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.walkers import cmu_humanoid

from mocapact.envs import dm_control_wrapper

class MocapTrackingGymEnv(dm_control_wrapper.DmControlWrapper):
    """
    Wraps the MultiClipMocapTracking into a Gym env.
    Adapted from https://github.com/microsoft/MoCapAct/blob/main/mocapact/envs/tracking.py
    """

    def __init__(
        self,
        dataset: str = None,
        ref_steps: Tuple[int] = (0,),
        mocap_path: Optional[Union[str, Path]] = None,
        task_kwargs: Optional[Dict[str, Any]] = None,
        environment_kwargs: Optional[Dict[str, Any]] = None,
        act_noise: float = 0.01,
        enable_all_proprios: bool = False,
        enable_cameras: bool = False,
        include_clip_id: bool = False,
        display_ghost: bool = True,

        # for rendering
        width: int = 640,
        height: int = 480,
        camera_id: int = 3
    ):
        if dataset is None:
            self._dataset = types.ClipCollection(ids=['CMU_002_01', 'CMU_009_01', 'CMU_010_04', 'CMU_013_11', 'CMU_014_06', 'CMU_041_02',
                        'CMU_046_01', 'CMU_075_01', 'CMU_083_18', 'CMU_105_53', 'CMU_143_41', 'CMU_049_07'])
        else:
            self._dataset = types.ClipCollection(ids=[dataset])
        self._enable_all_proprios = enable_all_proprios
        self._enable_cameras = enable_cameras
        self._include_clip_id = include_clip_id
        task_kwargs = task_kwargs or dict()
        task_kwargs['ref_path'] = mocap_path if mocap_path else cmu_mocap_data.get_path_for_cmu(version='2020')
        task_kwargs['dataset'] = self._dataset
        task_kwargs['ref_steps'] = ref_steps
        if display_ghost:
            task_kwargs['ghost_offset'] = np.array([1., 0., 0.])
        super().__init__(
            tracking.MultiClipMocapTracking,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            act_noise=act_noise,
            width=width,
            height=height,
            camera_id=camera_id
        )

    def _get_walker(self):
        return cmu_humanoid.CMUHumanoidPositionControlledV2020

    def _create_env(
        self,
        task_type,
        task_kwargs,
        environment_kwargs,
        act_noise=0.,
        arena_size=(8., 8.)
    ):
        env = super()._create_env(task_type, task_kwargs, environment_kwargs, act_noise, arena_size)
        walker = env._task._walker
        # Remove the contacts.
        # for geom in walker.mjcf_model.find_all('geom'):
        #     # alpha=0.999 ensures grey ghost reference.
        #     # for alpha=1.0 there is no visible difference between real walker and
        #     # ghost reference.
        #     alpha = 0.999
        #     if geom.rgba is not None and geom.rgba[3] < alpha:
        #         alpha = geom.rgba[3]

        #     geom.set_attributes(
        #         # contype=0,
        #         # conaffinity=0,
        #         rgba=(0.5, 0.5, 0.5, alpha))
                    
        if self._enable_all_proprios:
            walker.observables.enable_all()
            walker.observables.prev_action.enabled = False # this observable is not implemented
            if not self._enable_cameras:
                # TODO: procedurally find the cameras
                walker.observables.egocentric_camera.enabled = False
                walker.observables.body_camera.enabled = False
            env.reset()
        return env

    def _create_observation_space(self) -> spaces.Dict:
        obs_spaces = dict()
        for k, v in self._env.observation_spec().items():
            if v.dtype == np.float64 and np.prod(v.shape) > 0:
                obs_spaces[k] = spaces.Box(
                    -np.infty,
                    np.infty,
                    shape=(np.prod(v.shape),),
                    dtype=np.float32
                )
            elif v.dtype == np.uint8:
                tmp = v.generate_value()
                obs_spaces[k] = spaces.Box(
                    v.minimum.item(),
                    v.maximum.item(),
                    shape=tmp.shape,
                    dtype=np.uint8
                )
            elif k == 'walker/clip_id' and self._include_clip_id:
                obs_spaces[k] = spaces.Discrete(len(self._dataset.ids))
        return spaces.Dict(obs_spaces)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        obs, reward, done, info = super().step(action)
        info['time_in_clip'] = obs['walker/time_in_clip'].item()
        info['start_time_in_clip'] = self._start_time_in_clip
        info['last_time_in_clip'] = self._last_time_in_clip
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = self.get_observation(time_step)
        self._start_time_in_clip = obs['walker/time_in_clip'].item()
        self._last_time_in_clip = self._env.task._last_step / (len(self._env.task._clip_reference_features['joints'])-1)
        return obs