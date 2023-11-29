import os.path as osp
import numpy as np
import tree
import mujoco

from typing import Any, Callable, Dict, Optional, Text, Tuple
from dm_env import TimeStep
from dm_control import composer
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.mocap import loader
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.tasks.reference_pose import utils
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.walkers import initializers
from dm_control.suite.wrappers import action_noise
from gym import core
from gym import spaces
from gym import core, spaces
from dm_control import suite
from dm_env import specs
import numpy as np


class StandInitializer(initializers.WalkerInitializer):
    def __init__(self, mode='random'):
        ref_path = cmu_mocap_data.get_path_for_cmu(version='2020')
        mocap_loader = loader.HDF5TrajectoryLoader(ref_path)
        self.mode = mode
        if mode=='fixed':
            clip_ids = ['CMU_040_12']
        elif mode=='random':
            clip_ids = ['CMU_002_01', 'CMU_009_01', 'CMU_010_04', 'CMU_013_11', 'CMU_014_06', 'CMU_041_02',
                        'CMU_046_01', 'CMU_075_01', 'CMU_083_18', 'CMU_105_53', 'CMU_143_41', 'CMU_049_07',]
        else:
            raise NotImplementedError()

        self._stand_features = []
        for clip_id in clip_ids:
            trajectory = mocap_loader.get_trajectory(clip_id)
            clip_reference_features = trajectory.as_dict()
            clip_reference_features = tracking._strip_reference_prefix(clip_reference_features, 'walker/')
            self._stand_features.append(tree.map_structure(lambda x: x, clip_reference_features))

    def initialize_pose(self, physics, walker, random_state):
        clip_index = random_state.randint(0, len(self._stand_features))
        if self.mode=='fixed':
            index = random_state.randint(0, 100)
        elif self.mode=='random':
            index = random_state.randint(0, 10)
        random_features = tree.map_structure(lambda x: x[index], self._stand_features[clip_index])
        utils.set_walker_from_features(physics, walker, random_features)

        # Add gaussain noise to the current velocities.
        if self.mode=='random':
            velocity, angular_velocity = walker.get_velocity(physics)
            walker.set_velocity(
                physics,
                velocity=random_state.normal(0, 0.1, size=3) + velocity,
                angular_velocity=random_state.normal(0, 0.1, size=3)+angular_velocity)

        mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)


def _spec_to_box(spec, dtype):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)

class DMCWrapper(core.Env):
    """
    from https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
    """
    def __init__(
            self,
            domain_name,
            task_name,
            task_kwargs=None,
            from_pixels=False,
            height=84,
            width=84,
            camera_id=0,
            frame_skip=1,
            environment_kwargs=None,
            channels_first=True
    ):
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first

        # create task
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs
        )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values(),
                np.float64
            )

        self._state_space = _spec_to_box(
            self._env.observation_spec().values(),
            np.float64
        )

        self.current_state = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def dm_env(self):
        return self._env

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._true_action_space

    @property
    def reward_range(self):
        return 0, self._frame_skip

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self.get_observation(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra['discount'] = time_step.discount
        return obs, reward, done, extra

    def reset(self):
        time_step = self._env.reset()
        return time_step

    def get_observation(self, time_step: TimeStep) -> Dict[str, np.ndarray]:
        dm_obs = time_step.observation
        obs = _flatten_obs(dm_obs)
        return obs


    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )

class CMUHumanoidGymWrapper(core.Env):
    """
    Wraps the dm_control environment and task into a Gym env. The task assumes
    the presence of a CMU position-controlled humanoid.
    Adapted from:
    https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
    """

    metadata = {"render.modes": ["rgb_array"], "videos.frames_per_second": 30}

    def __init__(
            self,
            task_type: Callable[..., composer.Task],
            task_kwargs: Optional[Dict[str, Any]] = None,
            environment_kwargs: Optional[Dict[str, Any]] = None,
            act_noise: float = 0.,
            arena_size: Tuple[float, float] = (8., 8.),

            # for rendering
            width: int = 640,
            height: int = 480,
            camera_id: int = 3
    ):
        """
        task_kwargs: passed to the task constructor
        environment_kwargs: passed to composer.Environment constructor
        """
        task_kwargs = task_kwargs or dict()
        environment_kwargs = environment_kwargs or dict()

        # create task
        self._env = self._create_env(
            task_type,
            task_kwargs,
            environment_kwargs,
            act_noise=act_noise,
            arena_size=arena_size
        )
        self._original_rng_state = self._env.random_state.get_state()

        # Set observation and actions spaces
        self._observation_space = self._create_observation_space()
        action_spec = self._env.action_spec()
        dtype = np.float32
        self._action_space = spaces.Box(
            low=action_spec.minimum.astype(dtype),
            high=action_spec.maximum.astype(dtype),
            shape=action_spec.shape,
            dtype=dtype
        )

        # set seed
        self.seed()

        self._height = height
        self._width = width
        self._camera_id = camera_id

    @staticmethod
    def make_env_constructor(task_type: Callable[..., composer.Task]):
        return lambda *args, **kwargs: CMUHumanoidGymWrapper(task_type, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def dm_env(self) -> composer.Environment:
        return self._env

    @property
    def observation_space(self) -> spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def np_random(self):
        return self._env.random_state

    def seed(self, seed: Optional[int] = None):
        if seed:
            srng = np.random.RandomState(seed=seed)
            self._env.random_state.set_state(srng.get_state())
        else:
            self._env.random_state.set_state(self._original_rng_state)
        return self._env.random_state.get_state()[1]

    def _create_env(
            self,
            task_type,
            task_kwargs,
            environment_kwargs,
            act_noise=0.,
            arena_size=(8., 8.)
    ) -> composer.Environment:
        walker = self._get_walker()
        arena = self._get_arena(arena_size)
        task = task_type(
            walker,
            arena,
            **task_kwargs
        )
        env = composer.Environment(
            task=task,
            **environment_kwargs
        )
        task.random = env.random_state  # for action noise
        if act_noise > 0.:
            env = action_noise.Wrapper(env, scale=act_noise / 2)

        return env

    def _get_walker(self):
        directory = osp.dirname(osp.abspath(__file__))
        initializer = StandInitializer()
        return cmu_humanoid.CMUHumanoidPositionControlledV2020(initializer=initializer)

    def _get_arena(self, arena_size):
        return floors.Floor(arena_size)

    def _create_observation_space(self) -> spaces.Dict:
        obs_spaces = dict()
        for k, v in self._env.observation_spec().items():
            if v.dtype == np.float64 and np.prod(v.shape) > 0:
                if np.prod(v.shape) > 0:
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
        return spaces.Dict(obs_spaces)

    def get_observation(self, time_step: TimeStep) -> Dict[str, np.ndarray]:
        dm_obs = time_step.observation
        obs = dict()
        for k in self.observation_space.spaces:
            if self.observation_space[k].dtype == np.uint8:  # image
                obs[k] = dm_obs[k].squeeze()
            else:
                obs[k] = dm_obs[k].ravel().astype(self.observation_space[k].dtype)
        return obs

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        time_step = self._env.step(action)
        reward = time_step.reward or 0.
        done = time_step.last()
        obs = self.get_observation(time_step)
        info = dict(
            internal_state=self._env.physics.get_state().copy(),
            discount=time_step.discount
        )
        return obs, reward, done, info

    def reset(self) -> Dict[str, np.ndarray]:
        time_step = self._env.reset()
        return self.get_observation(time_step)

    def render(
            self,
            mode: Text = 'rgb_array',
            height: Optional[int] = None,
            width: Optional[int] = None,
            camera_id: Optional[int] = None
    ) -> np.ndarray:
        assert mode == 'rgb_array', "This wrapper only supports rgb_array mode, given %s" % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)