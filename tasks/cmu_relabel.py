# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Try to maximize speed on x-y plane, adopted from https://github.com/microsoft/MoCapAct/blob/main/mocapact/tasks/velocity_control.py
"""
import collections
import numpy as np
from dm_control import composer
from dm_control.composer.observation import observable as dm_observable
from dm_control.locomotion.tasks.reference_pose import tracking
from trajectory.utils.relabel_humanoid import project_left, project_forward, project_height


class CMURelabelTask(composer.Task):
    """
    A task that requires the walker to track a randomly changing velocity.
    """

    def __init__(
            self,
            walker,
            arena,
            max_speed=4.5,
            reward_margin=0.75,
            physics_timestep=tracking.DEFAULT_PHYSICS_TIMESTEP,
            control_timestep=0.03,
            contact_termination=True,
            relabel_type="speed"
    ):
        self._walker = walker
        self._arena = arena
        self._walker.create_root_joints(self._arena.attach(self._walker))
        self._max_speed = max_speed
        self._reward_margin = reward_margin
        self._move_speed = 0.
        self._move_angle = 0.
        self._move_speed_counter = 0.
        self.relabel_type = relabel_type
        self.enabled_observables = []
        self.enabled_observables += self._walker.observables.proprioception
        self.enabled_observables += self._walker.observables.kinematic_sensors
        self.enabled_observables += self._walker.observables.dynamic_sensors
        self.enabled_observables.append(self._walker.observables.sensors_touch)
        self.enabled_observables.append(self._walker.observables.egocentric_camera)
        for observable in self.enabled_observables:
            observable.enabled = True

        self.set_timesteps(physics_timestep=physics_timestep, control_timestep=control_timestep)
        self._contact_termination = contact_termination

    @property
    def root_entity(self):
        return self._arena

    def _is_disallowed_contact(self, contact):
        set1, set2 = self._walker_nonfoot_geomids, self._ground_geomids
        return ((contact.geom1 in set1 and contact.geom2 in set2) or
                (contact.geom1 in set2 and contact.geom2 in set1))

    def should_terminate_episode(self, physics):
        del physics
        return self._failure_termination

    def get_discount(self, physics):
        del physics
        if self._failure_termination:
            return 0.
        else:
            return 1.

    def initialize_episode(self, physics, random_state):
        self._walker.reinitialize_pose(physics, random_state)

        self._failure_termination = False
        walker_foot_geoms = set(self._walker.ground_contact_geoms)
        walker_nonfoot_geoms = [
            geom for geom in self._walker.mjcf_model.find_all('geom')
            if geom not in walker_foot_geoms
        ]
        self._walker_nonfoot_geomids = set(physics.bind(walker_nonfoot_geoms).element_id)
        self._ground_geomids = set(physics.bind(self._arena.ground_geoms).element_id)

    def get_reward(self, physics):
        if self.relabel_type == "speed":
            sensor_vel = self._walker.observables.sensors_velocimeter(physics)
            reward = np.linalg.norm(sensor_vel)
        elif self.relabel_type == "x_vel":
            reward = self._walker.observables.sensors_velocimeter(physics)[0]
        elif self.relabel_type == "y_vel":
            reward = self._walker.observables.sensors_velocimeter(physics)[1]
        elif self.relabel_type == "forward":
            reward = project_forward(self._walker.observables.sensors_velocimeter(physics),
                                     self._walker.observables.world_zaxis(physics))
        elif self.relabel_type == "backward":
            reward = -project_forward(self._walker.observables.sensors_velocimeter(physics),
                                        self._walker.observables.world_zaxis(physics))
        elif self.relabel_type == "shift_left":
            reward = project_left(self._walker.observables.sensors_velocimeter(physics),
                                     self._walker.observables.world_zaxis(physics))
        elif self.relabel_type == "jump":
            reward = np.maximum(0, project_height(self._walker.observables.sensors_velocimeter(physics),
                                                  self._walker.observables.world_zaxis(physics)))
        elif self.relabel_type == "z_vel":
            reward = self._walker.observables.sensors_velocimeter(physics)[2]
        elif self.relabel_type == "negative_z_vel":
            reward = -self._walker.observables.sensors_velocimeter(physics)[2]
        elif self.relabel_type == "rotate_x":
            reward = self._walker.observables.sensors_gyro(physics)[0]
        elif self.relabel_type == "rotate_y":
            reward = self._walker.observables.sensors_gyro(physics)[1]
        elif self.relabel_type == "rotate_z":
            reward = self._walker.observables.sensors_gyro(physics)[2]
        else:
            raise NotImplementedError()
        return reward

    def before_step(self, physics, action, random_state):
        self._walker.apply_action(physics, action, random_state)

    def after_step(self, physics, random_state):
        self._failure_termination = False
        if self._contact_termination:
            for contact in physics.data.contact:
                if self._is_disallowed_contact(contact):
                    self._failure_termination = True
                    break
        self._move_speed_counter += 1
