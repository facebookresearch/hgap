from torch.utils.data import Dataset
import bisect
import h5py
import itertools
import collections
import numpy as np
from gym import spaces
import functools
from typing import Dict, Sequence, Text, Optional, Union
from stable_baselines3.common.running_mean_std import RunningMeanStd
from trajectory.utils.relabel_humanoid import get_speed, get_angular_vel, get_body_height, get_vel, get_left_vel, get_height_vel, get_forward_vel
import torch
import rlds
import tensorflow as tf
import tensorflow_datasets as tfds
from trajectory.tfds import mocap_utils
MULTIPLIER = 10


MULTI_CLIP_OBSERVABLES_SANS_ID = (
    'walker/actuator_activation',
    'walker/appendages_pos',
    'walker/body_height',
    'walker/end_effectors_pos',
    'walker/gyro_control',
    'walker/joints_pos',
    'walker/joints_vel',
    'walker/joints_vel_control',
    'walker/sensors_accelerometer',
    'walker/sensors_gyro',
    'walker/sensors_torque',
    'walker/sensors_touch',
    'walker/sensors_velocimeter',
    'walker/time_in_clip',
    'walker/velocimeter_control',
    'walker/world_zaxis',
    'walker/reference_rel_joints',
    'walker/reference_rel_bodies_pos_global',
    'walker/reference_rel_bodies_quats',
    'walker/reference_rel_bodies_pos_local',
    'walker/reference_ego_bodies_quats',
    'walker/reference_rel_root_quat',
    'walker/reference_rel_root_pos_local',
    'walker/reference_appendages_pos',
)

CMU_HUMANOID_OBSERVABLES = (
    'walker/actuator_activation',
    'walker/appendages_pos',
    'walker/body_height',
    'walker/end_effectors_pos',
    'walker/joints_pos',
    'walker/joints_vel',
    'walker/sensors_accelerometer',
    'walker/sensors_gyro',
    'walker/sensors_torque',
    'walker/sensors_touch',
    'walker/sensors_velocimeter',
    'walker/world_zaxis'
)

return_mean = {
    "none": 11.799545288085938,
    "speed": 8.622393608093262,
    "rotate_y": 0.791003942489624,
    "forward": 4.268054485321045,
    "backward": -4.270056247711182,
    "shift_left": 0.04919257014989853,
    "jump": 0.9864307641983032,
    "rotate_x": -0.0740567222237587,
    "rotate_z": -0.054631639271974564
}

return_std = {
    "none": 2.7906644344329834,
    "speed": 7.853872299194336,
    "rotate_y": 15.0303316116333,
    "forward": 8.739494323730469,
    "backward": 8.741517066955566,
    "shift_left": 3.811504364013672,
    "jump": 1.3811616897583008,
    "rotate_x": 3.5730364322662354,
    "rotate_z": 3.9460842609405518
}

reward_mean = {
    "none": 0.6940667629241943,
    "speed": 0.5072286128997803,
    "rotate_y": 0.046567633748054504,
    "forward": 0.251115620136261,
    "backward": -0.25101155042648315,
    "shift_left": 0.002932528965175152,
    "jump": 0.058018457144498825,
    "rotate_x": -0.004368575755506754,
    "rotate_z": -0.003207581816241145
}

reward_std = {
    "none": 0.16416609287261963,
    "speed": 0.4620329737663269,
    "rotate_y": 0.8839800953865051,
    "forward": 0.5141152143478394,
    "backward": 0.5142410397529602,
    "shift_left": 0.22409744560718536,
    "jump": 0.08115938305854797,
    "rotate_x": 0.2102961540222168,
    "rotate_z": 0.23211251199245453
}



class SequentialDataLoader():
    def __init__(
            self,
            fnames: Union[Sequence[Text], Text],
            sequence_length,
            metrics_path: Text,
            batch_size: int,
            return_mean_act: bool = False,
            normalize_obs: bool = True,
            normalize_act: bool = True,
            normalize_reward: bool = False,
            observables: Union[Sequence[Text], Dict[Text, Sequence[Text]]] = CMU_HUMANOID_OBSERVABLES,
            concat_observables: bool = True,
            discount: float = 0.99,
            relabel_type: str = "none",
            reward_clip="none",
            body_height_penalty: float = 0.0,
            body_height_limit: float = 0.5,
            repeat: bool = True,
            shuffle_buffer: bool = True,
            deterministic: bool = False,
            need_to_extract_observables: bool = True,
            validation_episodes: int = 0,
            checkpoint_path: str = "",
            world_size: int = 1,
            rank: int = 0,
            ignore_incomplete_episodes: bool = False,
            load_percentage: int = 100,
        ):
        self.metrics_path = metrics_path
        if isinstance(fnames, Sequence) and "hdf5" in fnames[0]:
            reader = mocap_utils.MocapActHDF5DataSource(
                fnames,
                metrics_path=metrics_path,
                normalize_actions=normalize_act,
                normalize_observations=normalize_obs,
                observables=observables,
                concat_observations=concat_observables,
                mean_actions=return_mean_act,
                need_to_extract_observables=need_to_extract_observables
            )
            dataset = reader.make_episode_dataset(shuffle_files=not deterministic)
            self.need_to_extract_observables = need_to_extract_observables
            self.observable_indices = reader._observable_indices
        elif isinstance(fnames, str) and "mocapact" in fnames:
            if fnames == "mocapact-compact":
                name = "mocapact/small_cmu_observable"
            elif fnames == "mocapact-large-compact":
                name = "mocapact/large_cmu_observable"
            else:
                raise ValueError("Unknown dataset name.")
            # This splits the source at reading level which is more efficient than 
            # doing a ds.shard() later because then the records have been decoded.
            subsplit = tfds.even_splits("train[{}:]".format(validation_episodes), n=world_size, drop_remainder=False)[rank]
            reader = mocap_utils.MocapActTFDSDataSource(
                # TODO(yl): We should consider moving the normalizing and action choosing
                # logic to be shared between the TFDS and HDF5 loader.
                name=name,
                # Specify a split to retrieve. Only subsets of train is available
                # as there are no explicit train/val splits from MocapAct.
                # split="train[{}:]".format(validation_episodes),
                split=subsplit,
                # Same as the HDF5 data source.
                normalize_observations=normalize_obs,
                normalize_actions=normalize_act,
                use_mean_actions=return_mean_act,
            )
            dataset = reader.make_episode_dataset()
            self.need_to_extract_observables = False
        else:
            raise ValueError("Unknown data source type.")

        self.normalize_obs = normalize_obs
        self.normalize_act = normalize_act
        self.proprio_mean = reader.proprio_mean
        self.proprio_std = reader.proprio_std
        self.act_mean = reader.action_mean
        self.act_std = reader.action_std
        self.discount = discount
        self._observables = observables
        self.normalize_reward = normalize_reward
        self.reward_mean = reward_mean[relabel_type]
        self.reward_std = reward_std[relabel_type]
        self.return_mean = return_mean[relabel_type]
        self.return_std = return_std[relabel_type]

        dataset = self.pre_process(dataset, relabel_type, discount, sequence_length, deterministic, repeat, shuffle_buffer, batch_size,
                                   reward_clip, body_height_limit, body_height_penalty, ignore_incomplete_episodes=ignore_incomplete_episodes)
        self.data_loader = dataset.prefetch(tf.data.AUTOTUNE)
        self.data_iterator = self.data_loader.as_numpy_iterator()

        self.validation_episodes = validation_episodes
        if validation_episodes > 0:
            # This splits the source at reading level which is more efficient than 
            # doing a ds.shard() later because then the records have been decoded.
            subsplit = tfds.even_splits("train[:{}]".format(validation_episodes), n=world_size, drop_remainder=False)[rank]
            validation_reader = mocap_utils.MocapActTFDSDataSource(
                name=name,
                split=subsplit,
                normalize_observations=normalize_obs,
                normalize_actions=normalize_act,
                use_mean_actions=return_mean_act,
            )

            validation_dataset = validation_reader.make_episode_dataset()
            validation_dataset = self.pre_process(validation_dataset, relabel_type, discount, sequence_length, deterministic=True,
                                                  repeat=False, shuffle_buffer=True, batch_size=batch_size,
                                                  reward_clip=reward_clip, body_height_limit=body_height_limit,
                                                  body_height_penalty=body_height_penalty, ignore_incomplete_episodes=ignore_incomplete_episodes)
            self.validation_data_loader = validation_dataset.prefetch(tf.data.AUTOTUNE)
            self.validation_data_iterator = self.validation_data_loader.as_numpy_iterator()
            self.validation_set = [self.numpy_data_to_torch(validation_batch) for validation_batch in self.validation_data_iterator]
        else:
            self.validation_set = []

        self.checkpoint_path = checkpoint_path
        self._checkpoint = tf.train.Checkpoint(train=self.data_loader, validation=self.validation_data_loader)


    def pre_process(self, dataset, relabel_type, discount, sequence_length, deterministic, repeat, shuffle_buffer, batch_size,
                          reward_clip, body_height_limit, body_height_penalty, ignore_incomplete_episodes=False):
        denorm_func = self.denormalize_observations if self.normalize_obs else None
        dataset = dataset.map(
            functools.partial(relabel_reward, relabel_type=relabel_type, denormalize=denorm_func,
                              reward_clip=reward_clip, body_height_limit=body_height_limit, body_height_penalty=body_height_penalty),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.map(
            functools.partial(overwrite_value, discount=discount),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if self.normalize_reward:
            dataset = dataset.map(
                functools.partial(normalize_reward_return, reward_mean=self.reward_mean, return_mean=self.return_mean,
                                  reward_std=self.reward_std, return_std=self.return_std), num_parallel_calls=tf.data.AUTOTUNE)

        dataset: tf.data.Dataset = dataset.interleave(
            lambda episode: rlds.transformations.batch(
                episode["steps"], size=sequence_length, shift=1, drop_remainder=ignore_incomplete_episodes
            ),
            deterministic=deterministic,
            num_parallel_calls=tf.data.AUTOTUNE,
            cycle_length=16,
            block_length=16
        )
        dataset = dataset.map(
            functools.partial(mocap_utils.pad_steps, max_len=sequence_length)
        )
        if repeat:
            dataset = dataset.repeat()
        if shuffle_buffer:
            dataset = dataset.shuffle(100000, reshuffle_each_iteration=True, seed=0 if deterministic else None)

        dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset


    @property
    def observation_dim(self):
        return self.data_loader.element_spec["observation"].shape[-1]

    @property
    def action_dim(self):
        return self.data_loader.element_spec["action"].shape[-1]

    def numpy_data_to_torch(self, numpy_data):
        observation = torch.tensor(numpy_data["observation"], dtype=torch.float32)
        action = torch.tensor(numpy_data["action"], dtype=torch.float32)
        reward = torch.tensor(numpy_data["reward"], dtype=torch.float32)[..., None]
        value = torch.tensor(numpy_data["value"], dtype=torch.float32)[..., None]
        terminal = torch.tensor(numpy_data["is_terminal"], dtype=torch.float32)[..., None]
        terminal = 1 - torch.cumprod(1 - terminal, dim=1)
        mask = torch.tensor(numpy_data["mask"], dtype=torch.float32)[..., None]
        # also mask out if the actions are all zero (hacky way to deal with padding)
        mask *= (torch.logical_not(torch.all((action == 0), dim=-1, keepdim=True))).float()
        joined = torch.tensor(np.concatenate([observation, action, reward, value], axis=-1), dtype=torch.float32)
        return joined, mask, terminal

    def __iter__(self):
        return self

    def __next__(self):
        numpy_data = next(self.data_iterator)
        joined, mask, terminal = self.numpy_data_to_torch(numpy_data)
        return joined, mask, terminal

    def _extract_observations(self, all_obs: np.ndarray, observable_keys: Sequence[Text]):
            return {k: all_obs[..., self.observable_indices[k]] for k in observable_keys}

    def denormalize_reward(self, rewards):
        return rewards * self.reward_std + self.reward_mean

    def denormalize_return(self, returns):
        return returns * self.return_std + self.return_mean

    def normalize_observations(self, states):
        states_std = np.squeeze(np.array(self.proprio_std))
        states_mean = np.squeeze(np.array(self.proprio_mean))

        if self.need_to_extract_observables:
            states_std = self._extract_observations(states_std, self._observables)
            states_mean = self._extract_observations(states_mean, self._observables)
            states_std = np.concatenate(list(states_std.values()), axis=-1)
            states_mean = np.concatenate(list(states_mean.values()), axis=-1)

        if torch.is_tensor(states):
            states_std = torch.Tensor(states_std).to(states.device)
            states_mean = torch.Tensor(states_mean).to(states.device)
        return (states - states_mean) / states_std

    def denormalize_observations(self, observations):
        states_std = np.squeeze(np.array(self.proprio_std))
        states_mean = np.squeeze(np.array(self.proprio_mean))

        if self.need_to_extract_observables:
            obs_std = self._extract_observations(states_std, self._observables)
            obs_mean = self._extract_observations(states_mean, self._observables)
            obs_std = np.concatenate(list(obs_std.values()), axis=-1)
            obs_mean = np.concatenate(list(obs_mean.values()), axis=-1)
        else:
            obs_std = states_std
            obs_mean = states_mean

        if torch.is_tensor(observations):
            obs_std = torch.Tensor(states_std).to(observations.device)
            obs_mean = torch.Tensor(states_mean).to(observations.device)
        return observations * obs_std + obs_mean

    def denormalize_states(self, states):
        if torch.is_tensor(states):
            act_std = torch.Tensor(self.proprio_std).to(states.device)
            act_mean = torch.Tensor(self.proprio_std).to(states.device)
        else:
            act_std = np.squeeze(np.array(self.proprio_std))
            act_mean = np.squeeze(np.array(self.proprio_mean))
        states = states * act_std + act_mean
        return states

    def denormalize_actions(self, actions):
        if torch.is_tensor(actions):
            act_std = torch.Tensor(self.act_std).to(actions.device)
            act_mean = torch.Tensor(self.act_mean).to(actions.device)
        else:
            act_std = np.squeeze(np.array(self.act_std))
            act_mean =  np.squeeze(np.array(self.act_mean))
        actions = actions*act_std + act_mean
        return actions

    def denormalize_joined(self, joined):
        states = joined[:, :self.observation_dim]
        actions = joined[:, self.observation_dim:self.observation_dim+self.action_dim]
        left = joined[:, self.observation_dim+self.action_dim:]
        states = self.denormalize_observations(states)
        actions = self.denormalize_actions(actions)
        return np.concatenate([states, actions, left], axis=-1)
    
    def save(self):
        try:
            self._checkpoint.write(self.checkpoint_path)
        except Exception as e:
            print(f" {type(e)} \n Unable to checkpoint data loader.")
    
    def restore(self):
        try:
            self._checkpoint.read(self.checkpoint_path).assert_consumed()
            self.data_iterator = self.data_loader.as_numpy_iterator()
            self.validation_data_iterator = self.validation_data_loader.as_numpy_iterator()
            self.validation_set = [self.numpy_data_to_torch(validation_batch) for validation_batch in self.validation_data_iterator]
        except Exception as e:
            print(f" {type(e)} \n Unable to load data loader checkpoint.")


# TODO: create mapping between tasks and clips
TASK_MAPPING = {}

def get_clip_id(name):
    # suppose the name is of forward: /path/to/file/CMU_143_08.hdf5
    splitted = name.split('/')[-1].split('_')
    # print(splitted)
    assert len(splitted) == 3 and splitted[0] == 'CMU'
    return '_'.join(splitted[:2])


class MocapactDataset(Dataset):
    def __init__(
            self,
            hdf5_fnames: Sequence[Text],
            sequence_length,
            metrics_path: Text,
            return_mean_act: bool = False,
            normalize_obs: bool = True,
            normalize_act: bool = True,
            observables: Union[Sequence[Text], Dict[Text, Sequence[Text]]]=CMU_HUMANOID_OBSERVABLES,
            concat_observables: bool = True,
            keep_hdf5s_open: bool = False,
            relabel_type: str = "none",
            subset: Optional[Sequence[str]] = None,
            batch_size: int = None,
        ):
        """
                hdf5_fnames: List of paths to HDF5 files to load.
                observables: What observables to return in __getitem__.
                metrics_path: The path used to load the dataset metrics.
        """
        self._hdf5_fnames = hdf5_fnames
        if subset is not None or len(subset) > 0:
            self._subset_hdf5_names = self._get_subset(subset)
        self._observables = observables
        self._concat_observables = concat_observables
        self._min_seq_steps = -1
        self._max_seq_steps = sequence_length
        self.normalize_obs = normalize_obs
        self.normalize_act = normalize_act

        self._keep_hdf5s_open = keep_hdf5s_open
        if self._keep_hdf5s_open:
            self._dsets = [h5py.File(fname, 'r') for fname in self._hdf5_fnames]

        self._clip_snippets = []
        for fname in self._hdf5_fnames:
            try:
                with h5py.File(fname, 'r') as dset:
                    self._clip_snippets.append(tuple([k for k in dset.keys() if k.startswith('CMU')]))
            except:
                print('Error in file: ', fname)
            # remote file name from hdf5_fnames
            # self._hdf5_fnames.remove(fname)
        self._clip_snippets_flat = tuple(itertools.chain.from_iterable(self._clip_snippets))
        self._clip_ids = tuple({k.split('-')[0] for k in self._clip_snippets_flat})
        self._return_mean_act = return_mean_act
        self._metrics_path = metrics_path

        self.test_portion = 0.0

        with h5py.File(self._hdf5_fnames[0], 'r') as dset:
            self._ref_steps = dset['ref_steps'][...]
            obs_ind_dset = dset['observable_indices/walker']
            self.observable_indices = {
                f"walker/{k}": obs_ind_dset[k][...] for k in obs_ind_dset
            }
        self._set_spaces()
        self._set_stats()

        self._create_offsets()
        self.validation_episodes = 0
        self.validation_set = []

    @property
    def observation_dim(self):
        return self._observation_space.shape[0]

    @property
    def action_dim(self):
        return self._action_space.shape[0]

    def normalize_joined_single(self, joined):
        return joined
    
    def _get_subset(self, subset):
        """
        subset should be a list of task names defined in TASK_MAPPING, or clip prefix e.g. CMU_001
        """
        subset_clip_ids = []
        for s in subset:
            if s in TASK_MAPPING:
                subset_clip_ids.extend(TASK_MAPPING[s]) 
            else:
                assert 'CMU_' in s
                subset_clip_ids.append(s)
        self._hdf5_fnames = [name for name in self._hdf5_fnames if get_clip_id(name) in subset_clip_ids]
        print(f'\n[ datasets/mocapact ] loaded subset hdf5 files: {self._hdf5_fnames}\n')

    def _set_spaces(self):
        """
        Sets the observation and action spaces.
        """
        # Observation space for all observables in the dataset
        obs_spaces = {
            k: spaces.Box(-np.infty, np.infty, shape=v.shape, dtype=np.float32)
            for k, v in self.observable_indices.items()
        }
        self._full_observation_space = spaces.Dict(obs_spaces)

        # Observation space for the observables we're considering
        def make_observation_space(observables):
            observation_indices = {k: self.observable_indices[k] for k in observables}
            if self._concat_observables:
                observation_indices = np.concatenate(list(observation_indices.values()))
                return spaces.Box(low=-np.infty, high=np.infty, shape=observation_indices.shape)
            return spaces.Dict({
                observable: spaces.Box(
                    low=-np.infty,
                    high=np.infty,
                    shape=indices.shape,
                    dtype=np.float32
                ) for observable, indices in observation_indices.items()
            })
        if isinstance(self._observables, collections.abc.Sequence): # observables is Sequence[Text]
            self._observation_space = make_observation_space(self._observables)
        else: # observables is Dict[Text, Sequence[Text]]
            self._observation_space = {
                k: make_observation_space(subobservables) for k, subobservables in self._observables.items()
            }

        # Action space
        with h5py.File(self._hdf5_fnames[0], 'r') as dset:
            self._action_space = spaces.Box(
                low=np.float32(-1.),
                high=np.float32(1.),
                shape=dset[f"{self._clip_snippets[0][0]}/0/actions"].shape[1:],
                dtype=np.float32
            )

    def _extract_observations(self, all_obs: np.ndarray, observable_keys: Sequence[Text]):
        return {k: all_obs[..., self.observable_indices[k]] for k in observable_keys}

    def _set_stats(self):
        metrics = np.load(self._metrics_path, allow_pickle=True)
        self._count = metrics['count']
        self.proprio_mean = metrics['proprio_mean']
        self.proprio_var = metrics['proprio_var']
        if self._return_mean_act:
            self.act_mean = metrics['mean_act_mean']
            self.act_var = metrics['mean_act_var']
        else:
            self.act_mean = metrics['act_mean']
            self.act_var = metrics['act_var']
        self.snippet_returns = metrics['snippet_returns'].item()
        self.advantages = {k: v for k, v in metrics['advantages'].item().items() if k in self._clip_snippets_flat}
        self.values = {k: v for k, v in metrics['values'].item().items() if k in self._clip_snippets_flat}

        self.proprio_std = (np.sqrt(self.proprio_var) + 1e-4).astype(np.float32)
        self.act_std = (np.sqrt(self.act_var) + 1e-4).astype(np.float32)

        # Put observation statistics into RunningMeanStd objects
        self.obs_rms = dict()
        with h5py.File(self._hdf5_fnames[0], 'r') as dset:
            for k in dset['observable_indices/walker'].keys():
                key = "walker/" + k
                obs_rms = RunningMeanStd()
                obs_rms.mean = self.proprio_mean[self.observable_indices[key]]
                obs_rms.var = self.proprio_var[self.observable_indices[key]]
                obs_rms.count = self._count
                self.obs_rms[key] = obs_rms

        snippet_returns = np.array(list(self.snippet_returns.values()))
        advantages, values = [np.concatenate(list(x.values())) for x in [self.advantages, self.values]]
        self._return_offset = self._compute_offset(snippet_returns)
        self._advantage_offset = self._compute_offset(advantages)
        self._q_value_offset = self._compute_offset(values + advantages)

    def _compute_offset(self, array: np.ndarray):
        """
        Used to ensure the average data weight is approximately one.
        """
        return 0.

    def _create_offsets(self):
        self._total_len = 0
        self._dset_indices = []
        self._logical_indices, self._dset_groups = [[] for _ in self._hdf5_fnames], [[] for _ in self._hdf5_fnames]
        self._snippet_len_weights = [[] for _ in self._hdf5_fnames]
        iterator = zip(
            self._hdf5_fnames,
            self._clip_snippets,
            self._logical_indices,
            self._dset_groups,
            self._snippet_len_weights
        )
        for fname, clip_snippets, logical_indices, dset_groups, snippet_len_weights in iterator:
            with h5py.File(fname, 'r') as dset:
                self._dset_indices.append(self._total_len)
                dset_start_rollouts = dset['n_start_rollouts'][...]
                dset_rsi_rollouts = dset['n_rsi_rollouts'][...]
                n_start_rollouts = dset_start_rollouts
                n_rsi_rollouts = dset_rsi_rollouts
                for snippet in clip_snippets:
                    _, start, end = snippet.split('-')
                    clip_len = int(end)-int(start)
                    snippet_weight = 1

                    len_iterator = itertools.chain(
                        dset[f"{snippet}/start_metrics/episode_lengths"][:n_start_rollouts],
                        dset[f"{snippet}/rsi_metrics/episode_lengths"][:n_rsi_rollouts]
                    )
                    for i, ep_len in enumerate(len_iterator):
                        logical_indices.append(self._total_len)
                        dset_groups.append(f"{snippet}/{i if i < n_start_rollouts else i-n_start_rollouts+dset_start_rollouts}")
                        snippet_len_weights.append(snippet_weight)
                        if ep_len < self._min_seq_steps:
                            continue
                        self._total_len += snippet_weight * (ep_len - (self._min_seq_steps-1))

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        """
        TODO
        """
        dset_idx = bisect.bisect_right(self._dset_indices, idx)-1

        if self._keep_hdf5s_open:
            item = self._getitem(self._dsets[dset_idx], idx)
        else:
            with h5py.File(self._hdf5_fnames[dset_idx], 'r') as dset:
                item = self._getitem(dset, idx)
        return item
    
    def __getitems__(self, idxs):
        """
        More efficient way of loading batch items without excessive I/O
        """
        mapping = collections.defaultdict(list)
        for idx in idxs:
            dest_idx = bisect.bisect_right(self._dset_indices, idx)-1
            mapping[dest_idx].append(idx)
        items = []
        for dset_idx, d_idxs in mapping.items():
            if self._keep_hdf5s_open:
                for d_idx in d_idxs:
                    items.append((self._getitem(self._dsets[dset_idx], d_idx)))
            else:
                with h5py.File(self._hdf5_fnames[dset_idx], 'r') as dset:
                    for d_idx in d_idxs:
                            items.append((self._getitem(dset, d_idx)))
        return items

    def denormalize_rewards(self, rewards):
        return rewards * self._reward_std + self._reward_mean

    def normalize_observations(self, states):
        states_std = np.squeeze(np.array(self.proprio_std))
        states_mean = np.squeeze(np.array(self.proprio_mean))

        states_std = self._extract_observations(states_std, self._observables)
        states_mean = self._extract_observations(states_mean, self._observables)
        states_std = np.concatenate(list(states_std.values()), axis=-1)
        states_mean = np.concatenate(list(states_mean.values()), axis=-1)

        if torch.is_tensor(states):
            states_std = torch.Tensor(states_std).to(states.device)
            states_mean = torch.Tensor(states_mean).to(states.device)
        return (states - states_mean) / states_std

    def denormalize_observations(self, observations):
        states_std = np.squeeze(np.array(self.proprio_std))
        states_mean = np.squeeze(np.array(self.proprio_mean))

        obs_std = self._extract_observations(states_std, self._observables)
        obs_mean = self._extract_observations(states_mean, self._observables)
        obs_std = np.concatenate(list(obs_std.values()), axis=-1)
        obs_mean = np.concatenate(list(obs_mean.values()), axis=-1)

        if torch.is_tensor(observations):
            obs_std = torch.Tensor(states_std).to(observations.device)
            obs_mean = torch.Tensor(states_mean).to(observations.device)
        return observations * obs_std + obs_mean

    def denormalize_states(self, states):
        if torch.is_tensor(states):
            act_std = torch.Tensor(self.proprio_std).to(states.device)
            act_mean = torch.Tensor(self.proprio_std).to(states.device)
        else:
            act_std = np.squeeze(np.array(self.proprio_std))
            act_mean = np.squeeze(np.array(self.proprio_mean))
        states = states * act_std + act_mean
        return states

    def denormalize_actions(self, actions):
        if torch.is_tensor(actions):
            act_std = torch.Tensor(self.act_std).to(actions.device)
            act_mean = torch.Tensor(self.act_mean).to(actions.device)
        else:
            act_std = np.squeeze(np.array(self.act_std))
            act_mean =  np.squeeze(np.array(self.act_mean))
        actions = actions*act_std + act_mean
        return actions

    def denormalize_joined(self, joined):
        states = joined[:, :self.observation_dim]
        actions = joined[:, self.observation_dim:self.observation_dim+self.action_dim]
        left = joined[:, self.observation_dim+self.action_dim:]
        states = self.denormalize_observations(states)
        actions = self.denormalize_actions(actions)
        return np.concatenate([states, actions, left], axis=-1)

    def _getitem(self, dset, idx):
        dset_idx = bisect.bisect_right(self._dset_indices, idx)-1
        clip_idx = bisect.bisect_right(self._logical_indices[dset_idx], idx)-1
        act_name = "mean_actions" if self._return_mean_act else "actions"

        proprio_dset = dset[f"{self._dset_groups[dset_idx][clip_idx]}/observations/proprioceptive"]
        act_dset = dset[f"{self._dset_groups[dset_idx][clip_idx]}/{act_name}"]
        val_dset = dset[f"{self._dset_groups[dset_idx][clip_idx]}/values"]
        reward_dset = dset[f"{self._dset_groups[dset_idx][clip_idx]}/rewards"]
        snippet_len_weight = self._snippet_len_weights[dset_idx][clip_idx]

        start_idx = int((idx - self._logical_indices[dset_idx][clip_idx]) / snippet_len_weight)
        end_idx = min(start_idx + self._max_seq_steps, act_dset.shape[0]+1)
        all_obs = proprio_dset[start_idx:end_idx]
        act = act_dset[start_idx:end_idx]
        reward = reward_dset[start_idx:end_idx]
        value = val_dset[start_idx:end_idx]

        if self.normalize_obs:
            all_obs = (all_obs - self.proprio_mean) / self.proprio_std
        if self.normalize_act:
            act = (act - self.act_mean) / self.act_std

        # Extract observation
        if isinstance(self._observables, dict):
            obs = {
                k: self._extract_observations(all_obs, observable_keys)
                for k, observable_keys in self._observables.items()
            }
            if self._concat_observables:
                obs = {k: np.concatenate(list(v.values()), axis=-1) for k, v in obs.items()}
        else:
            obs = self._extract_observations(all_obs, self._observables)
            if self._concat_observables:
                obs = np.concatenate(list(obs.values()), axis=-1)

        padded_obs = np.pad(obs, ((0, self._max_seq_steps-obs.shape[0]), (0, 0)), constant_values=(0,0))
        padded_act = np.pad(act, ((0, self._max_seq_steps-act.shape[0]), (0, 0)), constant_values=(0,0))
        padded_reward = np.pad(reward[:, None], ((0, self._max_seq_steps-reward.shape[0]), (0, 0)), constant_values=(0,0))
        padded_value = np.pad(value[:, None], ((0, self._max_seq_steps-value.shape[0]), (0, 0)), constant_values=(0,0))
        joined = np.concatenate([padded_obs, padded_act, padded_reward, padded_value], axis=1)
        terminal = np.zeros_like(padded_value, dtype=np.float32)
        terminal[reward.shape[0]:] = 1
        mask = 1 - terminal
        return joined[:-1], joined[1:], mask[:-1], terminal[:-1]

def _discounted_return(rewards, discounts):
    def discounted_return_fn(acc, reward_discount):
        reward, discount = reward_discount
        return acc * discount + reward

    return tf.scan(
        fn=discounted_return_fn,
        elems=(rewards, discounts),
        reverse=True,
        initializer=0.0,
    )

def relabel_reward(episode, reward_clip, body_height_penalty, body_height_limit, relabel_type="speed", denormalize=None):
    if relabel_type == "none":
        return episode
    episode_length = episode["steps"].cardinality()
    steps = episode["steps"].batch(episode_length).get_single_element()
    observation = steps["observation"]
    if denormalize is not None:
        observation = denormalize(observation)
    if relabel_type == "speed":
        steps["reward"] = get_speed(observation)
    elif relabel_type == "x_vel":
        steps["reward"] = get_vel(observation, "x")
    elif relabel_type == "y_vel":
        steps["reward"] = get_vel(observation, "y")
    elif relabel_type == "forward":
        steps["reward"] = get_forward_vel(observation)
    elif relabel_type == "backward":
        steps["reward"] = -get_forward_vel(observation)
    elif relabel_type == "shift_left":
        steps["reward"] = get_left_vel(observation)
    elif relabel_type == "jump":
        steps["reward"] = tf.maximum(get_height_vel(observation), 0)
    elif relabel_type == "z_vel":
        steps["reward"] = get_vel(observation, "z")
    elif relabel_type == "negative_z_vel":
        steps["reward"] = -get_vel(observation, "z")
    elif relabel_type == "rotate_x":
        steps["reward"] = get_angular_vel(observation, "x")
    elif relabel_type == "rotate_y":
        steps["reward"] = get_angular_vel(observation, "y")
    elif relabel_type == "rotate_z":
        steps["reward"] = get_angular_vel(observation, "z")
    elif "tracking" in relabel_type:
        pass
    else:
        raise ValueError(f"Invalid relabel type: {relabel_type}")
    if body_height_penalty >= 0:
        steps["reward"] -= body_height_penalty * (tf.exp(tf.maximum(body_height_limit-get_body_height(observation), 0))-1)
    if reward_clip != "none":
        steps["reward"] = tf.clip_by_value(steps["reward"], -reward_clip, reward_clip)
    episode["steps"] = tf.data.Dataset.from_tensor_slices(steps)
    return episode

def overwrite_value(episode, discount: float):
    episode_length = episode["steps"].cardinality()
    steps = episode["steps"].batch(episode_length).get_single_element()
    returns = _discounted_return(steps["reward"], discount*tf.ones_like(steps["reward"]))
    # Scale RTG
    steps["value"] = returns
    episode["steps"] = tf.data.Dataset.from_tensor_slices(steps)
    return episode

def normalize_reward_return(episode, reward_mean, reward_std, return_mean, return_std):
    episode_length = episode["steps"].cardinality()
    steps = episode["steps"].batch(episode_length).get_single_element()
    steps["reward"] = (steps["reward"] - reward_mean) / reward_std
    steps["value"] = (steps["value"] - return_mean) / return_std
    episode["steps"] = tf.data.Dataset.from_tensor_slices(steps)
    return episode


if __name__ == "__main__":
    import os
    file_path = os.path.expanduser('~/data/mocap/dataset/small')
    file_names = [os.path.join(file_path, name) for name in os.listdir(file_path) if name.endswith('.hdf5')]
    data = MocapactDataset(file_names, 250, os.path.join(file_path, 'dataset_metrics.npz'))
    X, Y, mask, terminal = data[1000]
    print(X)
