# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import trajectory.utils as utils
import trajectory.datasets as datasets


def create_dataset(args, world_size=1, rank=0, repeat=True):
    sequence_length = args.subsampled_sequence_length * args.step
    
    if "large" in args.dataset:
        if os.path.exists(os.path.expanduser('~/data/mocap/dataset/large')):
            file_path = os.path.expanduser('~/data/mocap/dataset/large')
        else:
            file_path = os.path.expanduser('~/data_local/mocap/dataset/large')
    else:
        if os.path.exists(os.path.expanduser('~/data/mocap/dataset/small')):
            file_path = os.path.expanduser('~/data/mocap/dataset/small')
        else:
            file_path = os.path.expanduser('~/data_local/mocap/dataset/small')

    file_names = args.dataset 

    ignore_incomplete_episodes = args.ignore_incomplete_episodes

    dataset_config= utils.Config(
        datasets.SequentialDataLoader,
        batch_size=args.load_batch_size,
        savepath=(args.savepath, 'data_config.pkl'),
        fnames=file_names,
        normalize_obs=args.normalize,
        normalize_act=args.normalize,
        normalize_reward=args.normalize_reward,
        sequence_length=sequence_length,
        discount=args.discount,
        relabel_type=args.relabel_type,
        metrics_path=os.path.join(file_path, 'dataset_metrics.npz'),
        validation_episodes=args.validation_episodes,
        body_height_limit=args.body_height_limit,
        body_height_penalty=args.body_height_penalty,
        reward_clip=args.reward_clip,
        checkpoint_path=os.path.join(args.savepath, 'dataloader_checkpoint'),
        world_size=world_size,
        rank=rank,
        repeat=repeat,
        ignore_incomplete_episodes=ignore_incomplete_episodes,
    )

    dataset = dataset_config()
    return dataset
