import collections
import glob
from typing import Sequence, Union, Optional

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import rlds
from absl import logging
from dm_env import specs


gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

CMU_HUMANOID_OBSERVABLES = (
    "walker/actuator_activation",
    "walker/appendages_pos",
    "walker/body_height",
    "walker/end_effectors_pos",
    "walker/joints_pos",
    "walker/joints_vel",
    "walker/sensors_accelerometer",
    "walker/sensors_gyro",
    "walker/sensors_torque",
    "walker/sensors_touch",
    "walker/sensors_velocimeter",
    "walker/world_zaxis",
)


def _get_dataset_keys(h5file):
    """Gets the keys present in the D4RL dataset."""
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def _read_group(dataset_file):
    dataset_dict = {}
    for k in _get_dataset_keys(dataset_file):
        try:
            # first try loading as an array
            dataset_dict[k] = dataset_file[k][:]
        except ValueError:  # try loading as a scalar
            dataset_dict[k] = dataset_file[k][()]
    return dataset_dict


class MocapActTrajectoryReader:
    """Utility class for reading MocapAct HDF5 files."""

    def __init__(self, path: str) -> None:
        self._h5_file = h5py.File(path, mode="r")

    @property
    def h5_file(self):
        return self._h5_file

    def n_rsi_rollouts(self):
        return self._h5_file["n_rsi_rollouts"][()]

    def n_start_rollouts(self):
        return self._h5_file["n_start_rollouts"][()]

    def ref_steps(self):
        return self._h5_file["ref_steps"][:]

    def observable_indices(self):
        return _read_group(self._h5_file["observable_indices"])

    def stats(self):
        return _read_group(self._h5_file["stats"])

    def snippet_group_names(self):
        return [key for key in self._h5_file.keys() if key.startswith("CMU")]

    def get_snippet_start_metrics(self, snippet_name: str):
        return _read_group(self._h5_file[f"{snippet_name}/start_metrics"])

    def get_snippet_rsi_metrics(self, snippet_name: str):
        return _read_group(self._h5_file[f"{snippet_name}/rsi_metrics"])

    def get_snippet_early_termination(self, snippet_name: str):
        return self._h5_file[f"{snippet_name}/early_termination"][:]

    def get_snippet_num_episodes(self, snippet_name: str):
        return len(
            [key for key in self._h5_file[f"{snippet_name}"].keys() if key.isnumeric()]
        )

    def get_snippet_episode(self, snippet_name: str, episode_id: int):
        return _read_group(self._h5_file[f"{snippet_name}/{episode_id}"])

    def get_episode_from_key(self, key):
        return _read_group(self._h5_file[f"{key}"])

    def keys(self):
        indices = []
        snippet_groups = self.snippet_group_names()
        for snippet_name in snippet_groups:
            num_episodes = self.get_snippet_num_episodes(snippet_name)
            for episode_id in range(num_episodes):
                indices.append(f"{snippet_name}/{episode_id}")
        return indices


class MocapActHDF5DataSource:
    def __init__(
        self,
        pattern: Union[str, Sequence[str]],
        metrics_path: str,
        observables: Sequence[str] = CMU_HUMANOID_OBSERVABLES,
        concat_observations: bool = False,
        mean_actions: bool = False,
        normalize_observations: bool = False,
        normalize_actions: bool = False,
        need_to_extract_observables: bool = True,
    ) -> None:
        if isinstance(pattern, str):
            filenames = sorted(glob.glob(pattern))
        else:
            filenames = sorted(pattern)

        if not filenames:
            raise ValueError(f"Unable to find any files matching pattern {pattern}")

        self._filenames = filenames
        # NOTE(yl) the observables are sorted in a deterministic order, which
        # is different from what's implemented in transferplan.
        # This gives deterministic concat observations even when the order of
        # the observables are shuffled.
        self._observables = tuple(sorted(observables))
        self._concat_observations = concat_observations
        self._mean_actions = mean_actions
        self._normalize_observations = normalize_observations
        self._normalize_actions = normalize_actions
        self._need_to_extract_observables = need_to_extract_observables

        # Infer dataset spec using the first file in the pattern match
        reader = MocapActTrajectoryReader(filenames[0])
        self._observable_indices = reader.observable_indices()
        # Check that the observables requested is found in the dataset
        missing_observables = set(observables).difference(
            set(self._observable_indices.keys())
        )
        if missing_observables:
            raise ValueError(
                f"Missing observables {missing_observables} from the given files."
            )

        # Use an example episode to figure out the specs.
        episode_key = reader.keys()[0]
        logging.info("Infer dataset spec from %s (%s)", filenames[0], episode_key)
        example_episode = reader.get_episode_from_key(episode_key)
        self._set_element_spec(example_episode)
        self._set_environment_specs(example_episode)

        # Read the data metrics
        logging.debug("Read metrics from %s", metrics_path)
        self._set_metrics(metrics_path)
        self.proprio_mean = self._metrics["proprio_mean"]
        self.proprio_std = self._metrics["proprio_std"]
        self.action_mean = self._metrics["act_mean"]
        self.action_std = self._metrics["act_std"]

    def _set_element_spec(self, example_episode):
        actions = example_episode["actions"]
        observations = example_episode["observations/proprioceptive"]
        rewards = example_episode["rewards"]
        values = example_episode["values"]
        metadata_spec = {
            "episode_id": tf.TensorSpec(
                shape=(),
                dtype=tf.string,
            ),
            "episode_return": tf.TensorSpec(
                shape=(),
                dtype=tf.float32,
            ),
            "norm_episode_return": tf.TensorSpec(
                shape=(),
                dtype=tf.float32,
            ),
            "episode_length": tf.TensorSpec(
                shape=(),
                dtype=tf.float32,
            ),
            "norm_episode_length": tf.TensorSpec(
                shape=(),
                dtype=tf.float32,
            ),
            "early_termination": tf.TensorSpec(
                shape=(),
                dtype=tf.bool,
            ),
        }
        steps_spec = {
            "action": tf.TensorSpec(
                shape=(None,) + actions.shape[1:], dtype=actions.dtype
            ),
            "observation": tf.TensorSpec(
                shape=(None,) + observations.shape[1:],
                dtype=observations.dtype,
            ),
            "reward": tf.TensorSpec(
                shape=(None,) + rewards.shape[1:],
                dtype=rewards.dtype,
            ),
            "value": tf.TensorSpec(
                shape=(None,) + values.shape[1:],
                dtype=values.dtype,
            ),
        }

        self._element_spec = {
            "steps": steps_spec,
            **metadata_spec,
        }

    def _set_environment_specs(self, episode) -> None:
        observations = episode["observations/proprioceptive"]
        observation_indices = {
            k: self._observable_indices[k] for k in self._observables
        }
        if self._concat_observations:
            observation_indices = np.concatenate(list(observation_indices.values()))
            self._observation_spec = specs.Array(
                shape=observation_indices.shape, dtype=observations.dtype
            )
        else:
            self._observation_spec = collections.OrderedDict()
            for observable, indices in observation_indices.items():
                self._observation_spec[observable] = specs.Array(
                    shape=indices.shape, dtype=observations.dtype
                )
        actions = episode["actions"]
        rewards = episode["rewards"]
        self._action_spec = specs.BoundedArray(
            actions.shape[1:], dtype=actions.dtype, minimum=-1.0, maximum=1.0
        )
        self._reward_spec = specs.Array(rewards.shape[1:], dtype=rewards.dtype)
        self._discount_spec = specs.BoundedArray(
            rewards.shape[1:], dtype=np.float32, minimum=0, maximum=1.0
        )

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def reward_spec(self):
        return self._reward_spec

    def discount_spec(self):
        return self._discount_spec

    def _set_metrics(self, path):
        metrics_npz = np.load(path, allow_pickle=True)
        # Split proprio normalization stats based on observable_keys
        obs_mean = {}
        obs_std = {}
        for observable_name, indices in self._observable_indices.items():
            obs_mean[observable_name] = metrics_npz["proprio_mean"][..., indices]
            obs_std[observable_name] = np.sqrt(metrics_npz["proprio_var"][..., indices])
        self._metrics = {}

        if not self._need_to_extract_observables:
            # this generally means that we're using the compact dataset where the observations have been extracted, but metrics are not.
            observable_indices = np.concatenate(
                [
                    val
                    for key, val in self._observable_indices.items()
                    if key in CMU_HUMANOID_OBSERVABLES
                ],
                axis=-1,
            )
            self._metrics["proprio_mean"] = metrics_npz["proprio_mean"][
                ..., observable_indices
            ]
            self._metrics["proprio_std"] = (
                np.sqrt(metrics_npz["proprio_var"][..., observable_indices]) + 1e-4
            )
        else:
            self._metrics["proprio_mean"] = metrics_npz["proprio_mean"]
            self._metrics["proprio_std"] = np.sqrt(metrics_npz["proprio_var"]) + 1e-4
        self._metrics["obs_mean"] = obs_mean
        self._metrics["obs_std"] = obs_std

        self._metrics["act_mean"] = metrics_npz["act_mean"]
        self._metrics["act_std"] = np.sqrt(metrics_npz["act_var"]) + 1e-4
        self._metrics["mean_act_mean"] = metrics_npz["mean_act_mean"]
        self._metrics["mean_act_std"] = np.sqrt(metrics_npz["mean_act_var"])

        self._metrics["values"] = metrics_npz["values"].item()
        self._metrics["advantages"] = metrics_npz["advantages"].item()
        self._metrics["snippet_returns"] = metrics_npz["snippet_returns"].item()
        metrics_npz.close()

    def make_episode_dataset(
        self,
        shuffle_files: bool = True,
        num_parallel_calls: int = tf.data.AUTOTUNE,
    ) -> tf.data.Dataset:
        def episode_generator(path):
            reader = MocapActTrajectoryReader(path)
            # Using raw access to the HDF5 file is faster?
            # perhaps using the reader causes some GIL issues.
            h5_file = reader.h5_file
            n_rsi_rollouts = reader.n_rsi_rollouts()
            n_start_rollouts = reader.n_start_rollouts()
            snippet_names = reader.snippet_group_names()

            for snippet_name in snippet_names:
                rsi_metrics = _read_group(h5_file[f"{snippet_name}/rsi_metrics"])
                start_metrics = _read_group(h5_file[f"{snippet_name}/start_metrics"])
                early_terminations = h5_file[f"{snippet_name}/early_termination"][:]
                for episode_id in range(n_rsi_rollouts + n_start_rollouts):
                    key = f"{snippet_name}/{episode_id}"
                    # format: disable
                    if episode_id < n_rsi_rollouts:
                        stats = {
                            "episode_return": rsi_metrics["episode_returns"][
                                episode_id
                            ],
                            "norm_episode_return": rsi_metrics["norm_episode_returns"][
                                episode_id
                            ],
                            "episode_length": rsi_metrics["episode_lengths"][
                                episode_id
                            ],
                            "norm_episode_length": rsi_metrics["norm_episode_lengths"][
                                episode_id
                            ],
                            "early_termination": early_terminations[episode_id],
                        }
                    else:
                        i = episode_id - n_rsi_rollouts
                        stats = {
                            "episode_return": start_metrics["episode_returns"][i],
                            "norm_episode_return": start_metrics[
                                "norm_episode_returns"
                            ][i],
                            "episode_length": start_metrics["episode_lengths"][i],
                            "norm_episode_length": start_metrics[
                                "norm_episode_lengths"
                            ][i],
                            "early_termination": early_terminations[episode_id],
                        }
                    # format: enable
                    if self._mean_actions:
                        actions = h5_file[f"{key}/mean_actions"][:]
                    else:
                        actions = h5_file[f"{key}/actions"][:]
                    observations = h5_file[f"{key}/observations/proprioceptive"][:]
                    rewards = h5_file[f"{key}/rewards"]
                    values = h5_file[f"{key}/values"]
                    yield {
                        "steps": {
                            "observation": observations,
                            "action": actions,
                            "reward": rewards,
                            "value": values,
                        },
                        "episode_id": key,
                        **stats,
                    }

            h5_file.close()

        def maybe_normalize_steps(episode):
            steps = episode["steps"].copy()
            metadata = {k: v for k, v in episode.items() if k != "steps"}
            if self._normalize_observations:
                steps["observation"] = (
                    steps["observation"] - self._metrics["proprio_mean"]
                ) / self._metrics["proprio_std"]
            if self._normalize_actions:
                action_key = "mean_act" if self._mean_actions else "act"
                steps["action"] = (
                    steps["action"] - self._metrics[f"{action_key}_mean"]
                ) / self._metrics[f"{action_key}_std"]
            return {"steps": steps, **metadata}

        def extract_observations(episode):
            steps = episode["steps"].copy()
            metadata = {k: v for k, v in episode.items() if k != "steps"}
            flat_observations = steps["observation"]
            if self._need_to_extract_observables:
                if self._concat_observations:
                    observations = []
                    for observable in self._observables:
                        observations.append(
                            tf.gather(
                                flat_observations,
                                self._observable_indices[observable],
                                axis=-1,
                            )
                        )
                    observations = tf.concat(observations, axis=-1)
                else:
                    observations = collections.OrderedDict()
                    for index in self._observables:
                        observations[index] = tf.gather(
                            flat_observations, self._observable_indices[index], axis=-1
                        )
            else:
                if self._concat_observations:
                    observations = flat_observations
                else:
                    observations = collections.OrderedDict()
                    start_idx = 0
                    for index in self._observables:
                        observable_len = len(self._observable_indices[index])
                        observations[index] = flat_observations[
                            ..., start_idx : start_idx + observable_len
                        ]
                        start_idx += observable_len
            steps["observation"] = observations
            return {"steps": steps, **metadata}

        def convert_to_rlds_format(episode):
            metadata = {k: v for k, v in episode.items() if k != "steps"}

            def _pad_last(nest):
                return tf.nest.map_structure(
                    lambda x: tf.concat([x, tf.zeros_like(x[-1:])], axis=0), nest
                )

            steps = {
                "observation": episode["steps"]["observation"],
                "action": _pad_last(episode["steps"]["action"]),
                "reward": _pad_last(episode["steps"]["reward"]),
                "value": _pad_last(episode["steps"]["value"]),
                # Additional fields required by the RLDS dataset
                "is_first": tf.concat(
                    [
                        tf.ones(1, dtype=bool),
                        tf.zeros_like(episode["steps"]["reward"], bool),
                    ],
                    axis=0,
                ),
                "is_last": tf.concat(
                    [
                        tf.zeros_like(episode["steps"]["reward"], bool),
                        tf.ones(1, dtype=bool),
                    ],
                    axis=0,
                ),
                "is_terminal": tf.concat(
                    [
                        tf.zeros_like(episode["steps"]["reward"], bool),
                        tf.expand_dims(episode["early_termination"], 0),
                    ],
                    axis=0,
                ),
            }

            return {
                "steps": tf.data.Dataset.from_tensor_slices(steps),
                **metadata,
            }

        @tf.function
        def _fused_preprocess_fn(episode):
            return convert_to_rlds_format(
                extract_observations(maybe_normalize_steps(episode))
            )

        file_ds = tf.data.Dataset.from_tensor_slices(self._filenames)
        if shuffle_files:
            file_ds = file_ds.shuffle(
                len(self._filenames), reshuffle_each_iteration=True
            )
        dataset = file_ds.interleave(
            lambda f: tf.data.Dataset.from_generator(
                episode_generator, output_signature=self._element_spec, args=(f,)
            ),
            num_parallel_calls=num_parallel_calls,
            deterministic=not shuffle_files,
        )

        dataset = dataset.map(
            _fused_preprocess_fn, num_parallel_calls=num_parallel_calls
        )
        return dataset


class MocapActTFDSDataSource:
    def __init__(
        self,
        name: str,
        split: str = "train",
        observables: Sequence[str] = CMU_HUMANOID_OBSERVABLES,
        normalize_observations: bool = False,
        normalize_actions: bool = False,
        use_mean_actions: bool = False,
        concat_observations: bool = True,
        tfds_data_dir: Optional[str] = None,
    ):
        """Load Mocap trajectories from generated TFDS dataset."""
        from trajectory.tfds.tfds import mocapact as mocapact_tfds

        self._tfds_dataset, info = tfds.load(
            name,
            split=split,
            with_info=True,
            data_dir=tfds_data_dir,
        )
        metrics = info.metadata["metrics"]
        self._metrics = metrics
        self._normalize_observations = normalize_observations
        self._normalize_actions = normalize_actions
        self._use_mean_actions = use_mean_actions
        self._observables = tuple(sorted(observables))
        self._concat_observations = concat_observations
        self.set_mean_std()


    def set_mean_std(self):
        # Split proprio normalization stats based on observable_keys
        self.proprio_mean = np.concatenate([self._metrics["proprio_mean"][key] for key in self._observables])
        self.proprio_std = np.concatenate([self._metrics["proprio_std"][key] for key in self._observables])
        self.action_mean = self._metrics["act_mean"]
        self.action_std = self._metrics["act_std"]


    def make_episode_dataset(self):
        MEAN_ACTION = "mean_action"

        def normalize_steps(steps, normalize_observations, normalize_actions):
            if normalize_observations:
                observations = steps[rlds.OBSERVATION]
                normalized_observations = {}
                for key in steps[rlds.OBSERVATION]:
                    mean = self._metrics["proprio_mean"][key]
                    std = self._metrics["proprio_std"][key]
                    normalized_observations[key] = (observations[key] - mean) / (
                        std + 1e-4
                    )
                steps[rlds.OBSERVATION] = normalized_observations
            if normalize_actions:
                actions = steps[rlds.ACTION]
                mean_actions = steps[MEAN_ACTION]
                steps[rlds.ACTION] = (actions - self._metrics["act_mean"]) / (
                    self._metrics["act_std"]
                )
                steps[MEAN_ACTION] = (mean_actions - self._metrics["mean_act_mean"]) / (
                    self._metrics["mean_act_std"]
                )
            return steps

        def choose_actions(steps, use_mean_actions: bool):
            """Choose actions to be used."""
            steps = {k: v for k, v in steps.items()}
            mean_action = steps.pop(MEAN_ACTION)
            if use_mean_actions:
                steps[rlds.ACTION] = mean_action
            return steps

        def extract_observations(
            steps, observables: Sequence[str], concat_observations: bool
        ):
            if concat_observations:
                observations = tf.concat(
                    [steps[rlds.OBSERVATION][key] for key in observables], axis=-1
                )
            else:
                observations = {
                    key: steps[rlds.OBSERVATION][key] for key in observables
                }
            steps[rlds.OBSERVATION] = observations
            return steps

        def preprocess_steps(steps):
            steps = normalize_steps(
                steps,
                normalize_observations=self._normalize_observations,
                normalize_actions=self._normalize_actions,
            )
            steps = choose_actions(steps, use_mean_actions=self._use_mean_actions)
            steps = extract_observations(
                steps,
                observables=self._observables,
                concat_observations=self._concat_observations,
            )
            return steps

        def transform_episode(episode_dataset):
            cardinality = episode_dataset["steps"].cardinality()
            episode_dataset["steps"] = rlds.transformations.map_steps(
                episode_dataset["steps"], preprocess_steps
            ).apply(tf.data.experimental.assert_cardinality(cardinality))
            return episode_dataset

        dataset = self._tfds_dataset.map(
            transform_episode, num_parallel_calls=tf.data.AUTOTUNE
        )
        return dataset


def _pad_array_along_axis(x, padded_size, axis=0, value=0):
    pad_width = padded_size - tf.shape(x)[axis]
    if pad_width <= 0:
        return x
    padding = [(0, 0)] * len(x.shape.as_list())
    padding[axis] = (0, pad_width)
    padded = tf.pad(x, padding, mode="CONSTANT", constant_values=value)
    return padded


def _pad_along_axis(nest, padding_size, axis=0, value=0):
    return tf.nest.map_structure(
        lambda x: _pad_array_along_axis(x, padding_size, axis, value), nest
    )


def pad_steps(steps, max_len: int, add_mask: bool = True):
    """Pad batched steps of differnt length into the same size."""
    seq_len = tf.shape(tf.nest.flatten(steps)[0])[0]
    padded_steps = {}
    for k, v in steps.items():
        padded_steps[k] = _pad_along_axis(v, max_len, 0, 0)
    if add_mask:
        padded_steps["mask"] = _pad_along_axis(
            tf.ones((seq_len,), dtype=bool), max_len, 0, False
        )
    # Set shape to improve shape inference
    tf.nest.map_structure(
        lambda x: x.set_shape(
            [
                max_len,
            ]
            + x.shape[1:].as_list()
        ),
        padded_steps,
    )
    return padded_steps
