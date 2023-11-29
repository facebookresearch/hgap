"""mocapact dataset."""
from typing import List
import dataclasses
import os
import pickle
import tensorflow_datasets as tfds
import tensorflow as tf
import h5py
import numpy as np

from trajectory.tfds import mocap_utils

_DESCRIPTION = """
MoCapAct: A Multi-Task Dataset for Simulated Humanoid Control
"""
_CITATION = """
@inproceedings{wagener2022mocapact,
  title={{MoCapAct: A Multi-Task Dataset for Simulated Humanoid Control}},
  author={Wagener, Nolan and Kolobov, Andrey and Frujeri, Felipe Vieira and Loynd, Ricky and Cheng, Ching-An and Hausknecht, Matthew},
  booktitle={Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
"""

_OBSERVABLE_SHAPES = {
    "walker/actuator_activation": 56,
    "walker/appendages_pos": 15,
    "walker/body_height": 1,
    "walker/end_effectors_pos": 12,
    "walker/gyro_anticlockwise_spin": 1,
    "walker/gyro_backward_roll": 1,
    "walker/gyro_control": 3,
    "walker/gyro_rightward_roll": 1,
    "walker/head_height": 1,
    "walker/joints_pos": 56,
    "walker/joints_vel": 56,
    "walker/joints_vel_control": 56,
    "walker/orientation": 9,
    "walker/position": 3,
    "walker/reference_appendages_pos": 75,
    "walker/reference_ego_bodies_quats": 620,
    "walker/reference_rel_bodies_pos_global": 465,
    "walker/reference_rel_bodies_pos_local": 465,
    "walker/reference_rel_bodies_quats": 620,
    "walker/reference_rel_joints": 280,
    "walker/reference_rel_root_pos_local": 15,
    "walker/reference_rel_root_quat": 20,
    "walker/sensors_accelerometer": 3,
    "walker/sensors_gyro": 3,
    "walker/sensors_torque": 6,
    "walker/sensors_touch": 10,
    "walker/sensors_velocimeter": 3,
    "walker/time_in_clip": 1,
    "walker/torso_xvel": 1,
    "walker/torso_yvel": 1,
    "walker/veloc_forward": 1,
    "walker/veloc_strafe": 1,
    "walker/veloc_up": 1,
    "walker/velocimeter_control": 3,
    "walker/world_zaxis": 3,
}

# TODO(yl): Shared the processing with the HDF5 datasource.
def convert_to_rlds_format(episode):
    metadata = {k: v for k, v in episode.items() if k != "steps"}

    def _pad_last(nest):
        return tf.nest.map_structure(
            lambda x: np.concatenate([x, np.zeros_like(x[-1:])], axis=0), nest
        )

    steps = {
        "observation": episode["steps"]["observation"],
        "action": _pad_last(episode["steps"]["action"]),
        "mean_action": _pad_last(episode["steps"]["mean_action"]),
        "reward": _pad_last(episode["steps"]["reward"]),
        "value": _pad_last(episode["steps"]["value"]),
        # Additional fields required by the RLDS dataset
        "is_first": np.concatenate(
            [
                tf.ones(1, dtype=bool),
                tf.zeros_like(episode["steps"]["reward"], bool),
            ],
            axis=0,
        ),
        "is_last": np.concatenate(
            [
                np.zeros_like(episode["steps"]["reward"], bool),
                np.ones(1, dtype=bool),
            ],
            axis=0,
        ),
        "is_terminal": np.concatenate(
            [
                np.zeros_like(episode["steps"]["reward"], bool),
                np.expand_dims(episode["early_termination"], 0),
            ],
            axis=0,
        ),
    }

    return {
        "steps": steps,
        **metadata,
    }


# TODO(yl): Shared the processing with the HDF5 datasource.
def generate_episodes(path, observable_keys):
    reader = mocap_utils.MocapActTrajectoryReader(path)
    observable_indices = reader.observable_indices()
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
                    "episode_return": rsi_metrics["episode_returns"][episode_id],
                    "norm_episode_return": rsi_metrics["norm_episode_returns"][
                        episode_id
                    ],
                    "episode_length": rsi_metrics["episode_lengths"][episode_id],
                    "norm_episode_length": rsi_metrics["norm_episode_lengths"][
                        episode_id
                    ],
                    "early_termination": early_terminations[episode_id],
                }
            else:
                i = episode_id - n_rsi_rollouts
                stats = {
                    "episode_return": start_metrics["episode_returns"][i],
                    "norm_episode_return": start_metrics["norm_episode_returns"][i],
                    "episode_length": start_metrics["episode_lengths"][i],
                    "norm_episode_length": start_metrics["norm_episode_lengths"][i],
                    "early_termination": early_terminations[episode_id],
                }
            # format: enable
            mean_actions = h5_file[f"{key}/mean_actions"][:]
            actions = h5_file[f"{key}/actions"][:]
            flat_observations = h5_file[f"{key}/observations/proprioceptive"][:]
            observations = dict()
            for observable_name in observable_keys:
                idx = observable_indices[observable_name]
                observations[observable_name] = flat_observations[..., idx]
            rewards = h5_file[f"{key}/rewards"][:]
            values = h5_file[f"{key}/values"][:]
            yield key, {
                "steps": {
                    "observation": observations,
                    "action": actions,
                    "mean_action": mean_actions,
                    "reward": rewards,
                    "value": values,
                },
                "episode_id": key,
                **stats,
            }

    h5_file.close()


def _float_feature(shape, dtype, encoding=tfds.features.Encoding.ZLIB):
    return tfds.features.Tensor(shape=shape, dtype=dtype, encoding=encoding)


@dataclasses.dataclass
class MocapactBuilderConfig(tfds.core.BuilderConfig):
    """Configuration of the dataset generation process."""

    # Prefix in the download directory of MoCapAct
    # See https://github.com/microsoft/MoCapAct/blob/main/mocapact/download_dataset.py
    # Valid values are small/large
    prefix: str = "small"
    # Used for filtering observables used in the dataset (to reduce file size).
    observables: List[str] = mocap_utils.CMU_HUMANOID_OBSERVABLES


def _read_metrics(observable_indices, path: str):
    metrics_npz = np.load(path, allow_pickle=True)
    # Split proprio normalization stats based on observable_keys
    obs_mean = {}
    obs_std = {}
    metrics = {}

    for observable_name, indices in observable_indices.items():
        obs_mean[observable_name] = metrics_npz["proprio_mean"][..., indices]
        obs_std[observable_name] = np.sqrt(metrics_npz["proprio_var"][..., indices])

    metrics["proprio_mean"] = obs_mean
    metrics["proprio_std"] = obs_std

    metrics["act_mean"] = metrics_npz["act_mean"]
    metrics["act_std"] = np.sqrt(metrics_npz["act_var"]) + 1e-4
    metrics["mean_act_mean"] = metrics_npz["mean_act_mean"]
    metrics["mean_act_std"] = np.sqrt(metrics_npz["mean_act_var"])

    metrics["values"] = metrics_npz["values"].item()
    metrics["advantages"] = metrics_npz["advantages"].item()
    metrics["snippet_returns"] = metrics_npz["snippet_returns"].item()
    metrics_npz.close()
    return metrics


class MocapactMetadata(tfds.core.Metadata, dict):
    """MocapAct metrics saved as metadata"""

    def save_metadata(self, data_dir):
        """Save the metadata."""
        if "metrics" in self.keys():
            metrics_path = os.path.join(data_dir, "dataset_metrics.npz")
            with open(metrics_path, 'wb') as f:
                pickle.dump(self["metrics"], f)

    def load_metadata(self, data_dir):
        """Restore the metadata."""
        self.clear()
        metrics_path = os.path.join(data_dir, "dataset_metrics.npz")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                metrics = pickle.load(f)
            self.update({"metrics": metrics})


class Mocapact(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for mocapact dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    # NOTE: BUILDER_CONFIGS are used to specify the options
    # for building different processed datasets.
    # For now, only building the small dataset
    # with reduced CMU_HUMANOID observables is included.
    # To support additional configurations, append to the list by
    #   1. Change `prefix` to 'large' to build the large dataset.
    #   2. Change `observables` to all observables if you want to include more 
    #      observations.
    #   3. Give a unique name to the config so that we can use tfds.load
    #      for the different configuration.
    BUILDER_CONFIGS = [
        MocapactBuilderConfig(
            prefix="small",
            observables=mocap_utils.CMU_HUMANOID_OBSERVABLES,
            name="small_cmu_observable",
        ),
        MocapactBuilderConfig(
            prefix="large",
            observables=mocap_utils.CMU_HUMANOID_OBSERVABLES,
            name="large_cmu_observable",
        ),
    ]

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Download and put the MocapAct dataset in manual_dir
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(mocapact): Specifies the tfds.core.DatasetInfo object
        observable_keys = self.builder_config.observables

        # Compress observation and actions on disk.
        # There are probably no disk saving as these arrays are not
        # easily compressible. However, this will encode the tensors
        # in bytes which takes up less space compared to the default.
        observation_spec = tfds.features.FeaturesDict(
            {
                key: _float_feature(shape=(_OBSERVABLE_SHAPES[key],), dtype=tf.float32)
                for key in observable_keys
            }
        )
        action_spec = tfds.features.Tensor(
            shape=(56,), dtype=tf.float32, encoding=tfds.features.Encoding.ZLIB
        )
        steps_dict = {
            "observation": observation_spec,
            "action": action_spec,
            "mean_action": action_spec,
            "reward": tfds.features.Tensor(shape=(), dtype=tf.float32),
            "value": tfds.features.Tensor(shape=(), dtype=tf.float32),
            # Below are fields required by RLDS.
            "is_first": tfds.features.Tensor(shape=(), dtype=tf.bool),
            "is_last": tfds.features.Tensor(shape=(), dtype=tf.bool),
            "is_terminal": tfds.features.Tensor(shape=(), dtype=tf.bool),
        }
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "steps": tfds.features.Dataset(steps_dict),
                    "episode_id": tfds.features.Tensor(shape=(), dtype=tf.string),
                    "episode_return": tf.float32,
                    "norm_episode_return": tf.float32,
                    "episode_length": tf.int64,
                    "norm_episode_length": tf.int64,
                    "early_termination": tf.bool,
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # Set to `None` to disable
            homepage="https://microsoft.github.io/MoCapAct",
            citation=_CITATION,
            metadata=MocapactMetadata(),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.manual_dir
        metrics_path = path / "dataset_metrics.npz"
        # save metadata
        # TODO(yl): This is probably a bad way to save the metrics.
        # Figure out a better way to split the metrics.
        reference_filename = list(path.glob("*.hdf5"))[0]
        reader = mocap_utils.MocapActTrajectoryReader(reference_filename)
        observable_indices = reader.observable_indices()
        metrics = _read_metrics(observable_indices, metrics_path)
        self.info.metadata["metrics"] = metrics
        # TODO(mocapact): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(mocapact): Yields (key, example) tuples from the dataset
        for path in path.glob("*.hdf5"):
            for key, episode in generate_episodes(
                path, self.builder_config.observables
            ):
                yield key, convert_to_rlds_format(episode)


def _read_group(dataset_file):
    dataset_dict = {}
    for k in _get_dataset_keys(dataset_file):
        try:
            # first try loading as an array
            dataset_dict[k] = dataset_file[k][:]
        except ValueError:  # try loading as a scalar
            dataset_dict[k] = dataset_file[k][()]
    return dataset_dict


def _get_dataset_keys(h5file):
    """Gets the keys present in the D4RL dataset."""
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys
