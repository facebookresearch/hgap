import numpy as np
import tensorflow as tf
import torch

OBSERVABLES_DIM = [('actuator_activation', 56),
                   ('appendages_pos', 15),
                   ('body_height', 1),
                   ('end_effectors_pos', 12),
                   ('joints_pos', 56),
                   ('joints_vel', 56),
                   ('sensors_accelerometer', 3),
                   ('sensors_gyro', 3),
                   ('sensors_torque', 6),
                   ('sensors_touch', 10),
                   ('sensors_velocimeter', 3),
                   ('world_zaxis', 3)]

def get_observable_range(observable_name):
    """
    get the start and end index of the observable in the proprioceptive_obs
    """
    start = 0
    for name, dim in OBSERVABLES_DIM:
        if name == observable_name:
            return start, start + dim
        start += dim
    raise ValueError(f"observable_name {observable_name} not found")

def slice_observable(proprioceptive_obs, observable_name):
    """
    get the observable from proprioceptive_obs
    """
    start, end = get_observable_range(observable_name)
    return proprioceptive_obs[..., start:end]

def get_angular_vel(proprioceptive_obs, direction):
    """
    get the angular speed of the humanoid in from proprioceptive_obs.
    """
    angular_vel = slice_observable(proprioceptive_obs, 'sensors_gyro')
    if direction == 'x':
        angular_vel = angular_vel[..., 0]
    elif direction == 'y':
        angular_vel = angular_vel[..., 1]
    elif direction == 'z':
        angular_vel = angular_vel[..., 2]

    return angular_vel

def get_left_vel(proprioceptive_obs):
    """
    get the left speed of the humanoid in from proprioceptive_obs.
    cancel the sub-velocity towards world z axis according to the ego-centric world z axis.
    """
    ego_centric_vel = slice_observable(proprioceptive_obs, 'sensors_velocimeter')
    world_zaxis = slice_observable(proprioceptive_obs, 'world_zaxis')
    left_vel = project_left(ego_centric_vel, world_zaxis)
    return left_vel

def project_left(ego_centric_vel, world_zaxis):
    """
    project the velocity to the left direction
    """
    world_z_projected_to_ego_xy = world_zaxis[..., :-1]
    if isinstance(ego_centric_vel, tf.Tensor):
        world_z_normalized_to_ego_xy = tf.math.l2_normalize(world_z_projected_to_ego_xy, axis=-1)
    elif isinstance(ego_centric_vel, np.ndarray):
        world_z_normalized_to_ego_xy = world_z_projected_to_ego_xy / np.linalg.norm(world_z_projected_to_ego_xy, axis=-1, keepdims=True)
    elif isinstance(ego_centric_vel, torch.Tensor):
        world_z_normalized_to_ego_xy = torch.nn.functional.normalize(world_z_projected_to_ego_xy, dim=-1)
    else:
        raise ValueError(f"ego_centric_vel type {type(ego_centric_vel)} not supported")
    left_vel = world_z_normalized_to_ego_xy[..., 1] * ego_centric_vel[..., 0]
    return left_vel

def get_forward_vel(proprioceptive_obs):
    """
    get the forward speed of the humanoid in from proprioceptive_obs.
    cancel the sub-velocity towards world z axis according to the ego-centric world z axis.
    """
    ego_centric_vel = slice_observable(proprioceptive_obs, 'sensors_velocimeter')
    world_zaxis = slice_observable(proprioceptive_obs, 'world_zaxis')
    forward_vel = project_forward(ego_centric_vel, world_zaxis)
    return forward_vel

def project_forward(ego_centric_vel, world_zaxis):
    """
    project the velocity to the forward direction
    """
    world_z_projected_to_ego_yz = world_zaxis[..., 1:]
    if isinstance(ego_centric_vel, tf.Tensor):
        world_z_normalized_to_ego_yz = tf.math.l2_normalize(world_z_projected_to_ego_yz, axis=-1)
    elif isinstance(ego_centric_vel, np.ndarray):
        world_z_normalized_to_ego_yz = world_z_projected_to_ego_yz / np.linalg.norm(world_z_projected_to_ego_yz, axis=-1, keepdims=True)
    elif isinstance(ego_centric_vel, torch.Tensor):
        world_z_normalized_to_ego_yz = torch.nn.functional.normalize(world_z_projected_to_ego_yz, dim=-1)
    else:
        raise ValueError(f"ego_centric_vel type {type(ego_centric_vel)} not supported")
    forward_vel = world_z_normalized_to_ego_yz[..., 0] * ego_centric_vel[..., 2]
    return forward_vel

def get_height_vel(proprioceptive_obs):
    """
    get the height speed of the humanoid in from proprioceptive_obs.
    """
    ego_centric_vel = slice_observable(proprioceptive_obs, 'sensors_velocimeter')
    world_zaxis = slice_observable(proprioceptive_obs, 'world_zaxis')
    height_vel = project_height(ego_centric_vel, world_zaxis)
    return height_vel

def project_height(ego_centric_vel, world_zaxis):
    """
    project the velocity to the height direction
    """
    if isinstance(ego_centric_vel, tf.Tensor):
        height_vel = tf.reduce_sum(ego_centric_vel * world_zaxis, axis=-1)
    elif isinstance(ego_centric_vel, np.ndarray):
        height_vel = np.sum(ego_centric_vel * world_zaxis, axis=-1)
    elif isinstance(ego_centric_vel, torch.Tensor):
        height_vel = torch.sum(ego_centric_vel * world_zaxis, dim=-1)
    else:
        raise ValueError("ego_centric_vel should be either tf.Tensor or np.ndarray")
    return height_vel


def get_vel(proprioceptive_obs, direction):
    current_vel = slice_observable(proprioceptive_obs, 'sensors_velocimeter')
    if direction == 'x':
        vel = current_vel[..., 0]
    elif direction == 'y':
        vel = current_vel[..., 1]
    elif direction == 'z':
        vel = current_vel[..., 2]
    else:
        raise ValueError(f"direction {direction} not found")
    return vel


def get_body_height(proprioceptive_obs):
    """
    get the height of the humanoid in from proprioceptive_obs.
    """
    return slice_observable(proprioceptive_obs, 'body_height')[..., 0]

def get_speed(proprioceptive_obs):
    """
    get the speed of the humanoid in from proprioceptive_obs.
    Note that the speed is a non-negative scalar and do not indicate direction (different from velocity)
    """
    current_vel = slice_observable(proprioceptive_obs, 'sensors_velocimeter')
    if isinstance(current_vel, tf.Tensor):
        speed = tf.norm(current_vel, axis=-1)
    elif isinstance(current_vel, np.ndarray):
        speed = np.linalg.norm(current_vel, axis=-1)
    elif isinstance(current_vel, torch.Tensor):
        speed = torch.norm(current_vel, dim=-1)
    return speed


def get_x_speed(proprioceptive_obs):
    """
    get the speed of the humanoid in x direction from proprioceptive_obs.
    Note that the speed is a non-negative scalar and do not indicate direction (different from velocity)
    """
    current_vel = proprioceptive_obs[..., -6, None]
    if isinstance(current_vel, tf.Tensor):
        speed = tf.norm(current_vel, axis=-1)
    elif isinstance(current_vel, np.ndarray):
        speed = np.linalg.norm(current_vel, axis=-1)
    return speed

def get_x_negative(proprioceptive_obs, upper_bound=2.0):
    current_vel = proprioceptive_obs[..., -6]
    if isinstance(current_vel, tf.Tensor):
        reward = -tf.clip_by_value(current_vel, -upper_bound, upper_bound)
    elif isinstance(current_vel, np.ndarray):
        reward = -np.clip(current_vel, -upper_bound, upper_bound)
    return reward

def get_target_similarity(obs, target):
    """
    get the similarity between the current obs and the target with rbf kernel
    """
    current_vel = slice_observable(obs, 'sensors_velocimeter')
    if isinstance(obs, tf.Tensor):
        return tf.exp(-tf.norm(current_vel - tf.convert_to_tensor(target), axis=-1))
    elif isinstance(obs, np.ndarray):
        return np.exp(-np.linalg.norm(current_vel - target, axis=-1))
    else:
        raise ValueError("obs should be either tf.Tensor or np.ndarray")