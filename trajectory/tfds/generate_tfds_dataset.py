"""
Example for the new TFDS based dataset.

This shows how to use the new `MocapActTFDSDataSource` for loading
episodes from the MocapAct dataset.

Right now, the new interface does not yet implement inference of the example
specs like the old interface. However, this will be added in the near future.
Otherwise, the TFDS interface should provide identical episode tf.data.Dataset
like the old HDF5 data source.

The new TFDS interface uses https://www.tensorflow.org/datasets for converting
and processing the raw HDF5 into TFRecord which provides a more scalable pipeline
and addresses some issues with imbalanced loading from the old HDF5 pipeline.

You don't have to know TFDS in detail to use it. In particular, how to implement
the dataset builder in trajectory/jax/tap/mocap/tfds.

Having said that, it will be useful to know to use effectively use the new interface
if you spend some time reading through a basic tutorial, which can be found at

    https://www.tensorflow.org/datasets/overview#load_a_dataset

There are some additional details which would be useful to know when we want to
support building more customized datasets. The extension points have been
annotated with `NOTE`.

"""
import functools
import os

import rlds
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags

from trajectory.tfds import mocap_utils

# This is needed so that tfds can discover our mocapact builder.
from trajectory.tfds.tfds import mocapact

_MOCAPACT_DATA_DIR = flags.DEFINE_string(
    "mocapact_data_dir",
    default=os.environ.get("MOCAPACT_DATA_DIR", None),
    help="Path to the MocapAct download directory.",
)
_MOCAPACT_DATA_SIZE = flags.DEFINE_string(
    "size",
    default="small",
    help="Size of the MocapAct dataset.",
)

def main(_):
    mocap_data_dir = _MOCAPACT_DATA_DIR.value
    assert mocap_data_dir is not None
    # Use case 1. Manual way of preparing dataset
    # Prepare the dataset.
    data_size = _MOCAPACT_DATA_SIZE.value
    builder = tfds.builder(f"mocapact/{data_size}_cmu_observable")
    # NOTE: There's no downloading.
    # Instead, we specify the manual_dir that includes the original HDF5 dataset.
    # Overriding the manual_dir allows us to specify a custom location
    # for the original HDF5 files.
    # The manual_dir should point to the directory where the MocapAct dataset is
    # downloaded. For example, consider the download directory `./data` and
    # you have downloaded the small subset. Then the manual_dir would be `./data`
    # and the builder will look for HDF5 files located in `./data/dataset/small`
    # The later parts of the path can be customized with a BuilderConfig.
    # See mocap/tfds/mocapact/mocapact.py
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(manual_dir=mocap_data_dir)
    )
    # Use case 2. Use MocapActTFDSSource to retrieve episode dataset
    # similar to the HDF5 data source
    # Specify "mocapact" as the name will build the dataset with the default config
    # (small_cmu_observable).
    # Which is to build against the small/ directory with reduced CMU observables.
    # An alternative way of specifying the name is `mocapact/small_cmu_observable`.
    data_source = mocap_utils.MocapActTFDSDataSource(
        name="mocapact",
        # Specify a split to retrieve. Only subsets of train is available
        # as there are no explicit train/val splits from MocapAct.
        split="train",
        # Same as the HDF5 data source.
        normalize_observations=True,
        normalize_actions=True,
        use_mean_actions=True,
    )
    # Same as the HDF5 data source.
    episode_dataset = data_source.make_episode_dataset()
    print(episode_dataset.element_spec["steps"])
    sequence_length = 25
    # Same way to convert to the sequence dataset.
    sequence_dataset: tf.data.Dataset = episode_dataset.interleave(
        lambda episode: rlds.transformations.batch(
            episode["steps"], size=sequence_length, shift=1, drop_remainder=False
        ),
        deterministic=True,
        num_parallel_calls=tf.data.AUTOTUNE,
        cycle_length=16,
        block_length=16,
    )
    sequence_dataset = sequence_dataset.map(
        functools.partial(mocap_utils.pad_steps, max_len=sequence_length)
    )
    print(sequence_dataset.element_spec)


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    app.run(main)
