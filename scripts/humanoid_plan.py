# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import pdb
from os.path import join
from trajectory.utils.rendering import HumanoidRnederer
import torch
import os
import numpy as np
import imageio

from accelerate.logging import get_logger

os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
torch.backends.cuda.matmul.allow_tf32 = True

import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.search import (
    sample_with_prior,
    beam_with_prior,
    make_prefix,
    extract_actions,
)

class Parser(utils.Parser):
    dataset: str = 'mocapact'
    config: str = 'config.vqvae'


def main():
    #######################
    ######## setup ########
    #######################

    args = Parser().parse_args('plan')

    if "vae_name" in args.__dict__ and args.vae_name != "":
        vae_name = args.vae_name
    else:
        vae_name = args.exp_name
    print(args.vae_name)

    logger = get_logger(__name__, log_level="DEBUG")

    #######################
    ####### models ########
    #######################

    if "." in args.dataset:
        args.task = args.dataset
    if "task" not in args.__dict__ or args.task == "":
        args.task = Parser().parse_args('train').relabel_type

    env = datasets.load_environment(args.task)

    vae, gpt_epoch = utils.load_model(logger, args.logbase, args.dataset, vae_name,
                                    epoch=args.gpt_epoch, device=args.device)

    prior, _ = utils.load_transformer_model(logger, args.logbase, args.dataset, args.prior_name,
                                    epoch=args.gpt_epoch, device=args.device)

    if args.critic_name != "":
        critic, _ = utils.load_transformer_model(logger, args.logbase, args.dataset, args.critic_name,
                                                 epoch=args.gpt_epoch, device=args.device, type='critic')

    vae.set_padding_vector(np.zeros(vae.transition_dim - 1))
    #######################
    ####### dataset #######
    #######################
    renderer = HumanoidRnederer(datasets.load_environment(args.task), observation_dim=vae.observation_dim)


    if args.critic_name != "":
        dataset = utils.load_from_config(logger, args.logbase, args.dataset, args.critic_name,
                                         'data_config.pkl')
    else:
        dataset = utils.load_from_config(logger, args.logbase, args.dataset, args.prior_name,
                                         'data_config.pkl')

    timer = utils.timer.Timer()

    discount = dataset.discount
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    preprocess_fn = datasets.get_preprocess_fn(env.name)

    #######################
    ###### main loop ######
    #######################
    REWARD_DIM = VALUE_DIM = 1
    transition_dim = observation_dim + action_dim + REWARD_DIM + VALUE_DIM
    time_step = env.dm_env.reset()
    observation = env.get_observation(time_step)
    total_reward = 0
    discount_return = 0
    frames = []

    ## previous (tokenized) transitions for conditioning transformer
    context = []
    values = []

    T = 400
    vae.eval()
    for t in range(T):
        observation = preprocess_fn(observation)

        if dataset.normalize_obs:
            observation = dataset.normalize_observations(observation)

        if t % args.plan_freq == 0:
            ## concatenate previous transitions and current observations to input to model
            prefix = make_prefix(context, observation, transition_dim, device=args.device)[-1, -1, None, None]

            ## sample sequence from model beginning with `prefix`
            if args.test_planner == 'beam_with_prior':
                prior.eval()
                sequence = beam_with_prior(prior, vae, prefix,
                                           denormalize_rew=dataset.denormalize_reward,
                                           denormalize_val=dataset.denormalize_return,
                                           discount=args.discount,
                                           steps=args.horizon,
                                           optimize_target=args.objective,
                                           beam_width=args.beam_width,
                                           n_expand=args.n_expand)
            elif args.test_planner == 'beam_prior_perturb':
                prior.eval()
                sequence, value = beam_with_prior_perturb(prior, critic, vae, prefix,
                                                          args.horizon,
                                                          beam_width=args.beam_width,
                                                          n_expand=args.n_expand,
                                                          prob_threshold=args.prob_threshold,
                                                          ood_weight=args.prob_weight,
                                                          temperature=args.temperature,
                                                          normalize_value=critic.normalize_returns,
                                                          advantage_weight=args.advantage_weight,
                                                          )
            elif args.test_planner == "sample_with_prior":
                prior.eval()
                sequence = sample_with_prior(prior, vae, prefix, None, None, args.discount, args.horizon, nb_samples=args.nb_samples, rounds=1,
                                             likelihood_weight=args.prob_weight, prob_threshold=args.prob_threshold, top_p=args.top_p,
                                             temperature=args.temperature, objective=args.objective)
            else:
                raise NotImplementedError(f"Unknown planner type {args.test_planner}.")
        else:
            sequence = sequence[1:]

        ## [ horizon x transition_dim ] convert sampled tokens to continuous trajectory
        sequence_recon = sequence

        ## [ action_dim ] index into sampled trajectory to grab first action
        feature_dim = dataset.observation_dim
        action = extract_actions(sequence_recon, feature_dim, action_dim, t=0)
        if dataset.normalize_act:
            action = dataset.denormalize_actions(action)

        ## execute action in environment
        time_step = env.dm_env.step(action)
        next_observation = env.get_observation(time_step)
        reward = time_step.reward
        terminal = time_step.last()


        ## update return
        total_reward += reward
        discount_return += reward* discount**(t)

        img = env.render()
        frames.append(img)

        print(
            f'[ plan ] t: {t} / {T} | r: {reward:.2f} | R: {total_reward:.2f} | '
            f'time: {timer():.4f} | {args.dataset} | {args.exp_name} | {args.suffix}\n'
        )

        ## visualization
        if t % args.vis_freq == 0 or terminal or t == T-1:
            if not os.path.exists(args.savepath):
                os.makedirs(args.savepath)
        if terminal: break
        observation = next_observation

    imageio.mimsave(join(args.savepath, 'rollout.gif'), frames)
    ## save result as a json file
    json_path = join(args.savepath, 'rollout.json')
    json_data = {'step': t, 'return': float(total_reward), 'term': terminal, 'gpt_epoch': gpt_epoch, 'value_mean': float(np.mean(values)),
                'first_value': np.nan, 'first_search_value': np.nan, 'discount_return': float(discount_return),
                'prediction_error': np.nan}
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
