# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
import math
import torch
import numpy as np
import pickle
import os
import random
from torch.utils.data.dataloader import DataLoader
from trajectory.datasets.mocapact import SequentialDataLoader
from trajectory.utils.dataset import create_dataset
from trajectory.utils.scaling_law import fit_power_law
import torch.nn as nn
import wandb

from datetime import timedelta
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs
from accelerate.logging import get_logger

from timeit import default_timer as timer
import trajectory.utils as utils


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def to(xs, device):
    return [x.to(device) for x in xs]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _is_namedtuple(obj):
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    return (
        isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")
    )

def aggregate_loss(losses_and_log):
    result = []
    for loss_or_log in losses_and_log:
        if _is_namedtuple(loss_or_log):
            loss_or_log = loss_or_log._asdict()
        if isinstance(loss_or_log, dict):
            loss_or_log = {k: v.mean() for k,v in loss_or_log.items()}
        else:
            loss_or_log = loss_or_log.mean()
        result.append(loss_or_log)
    return tuple(result)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

import torch
import torch.profiler

def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))

class VQTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.n_tokens_target = config.n_tokens_target
        self.last_logging_time = 0

        self.optimizer = None

        self.logger = get_logger(__name__, log_level="DEBUG")

        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=18000))
        mixed_precision = 'fp16' if config.enable_fp16 else None
        self.accelerator = Accelerator(split_batches=True, even_batches=True, log_with="wandb", mixed_precision=mixed_precision, kwargs_handlers=[kwargs])

    def get_optimizer(self, model):
        if self.optimizer is None:
            self.accelerator.print(f'[ utils/training ] Making optimizer at epoch {self.stats["n_epochs"]}')
            if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
                self.optimizer = model.module.configure_optimizers(self.config)
            else:
                self.optimizer = model.configure_optimizers(self.config)
        return self.optimizer

    def init_data_loader(self, dataset, stats=None):

        config = self.config
        if not isinstance(dataset, SequentialDataLoader):
            g = torch.Generator()
            g = g.manual_seed(torch.initial_seed())

            # scale batch sizes according to number of gpus

            sampler = utils.BatchSampler(dataset, config.load_batch_size, generator=g)
            loader = DataLoader(
                dataset,
                pin_memory=True,
                batch_sampler=sampler,
                num_workers=config.num_workers,
                worker_init_fn=seed_worker,
                generator=g,
            )
            self.loader = iter(cycle(loader))
        else:
            self.loader = dataset
    
    def init_stats(self, stats):
        self.stats = stats
    
    def init_wandb(self, conf, name):
        self.accelerator.init_trackers("latentPlanning", init_kwargs={"wandb":{"name": name, **conf}})

    def train(self, model, dataset, log_freq=1e5, save_freq=0, stats=None, savepath=None, debug=False, validation_period=12):
        """
        train 1M tokens
        log_freq: number of seconds between logging
        validation_period: number of logging cycles between validation
        """

        config = self.config
        enable_fp16 = config.enable_fp16

        optimizer = self.get_optimizer(model)
        model.train(True)

        ## seed the dataloader to support interrupted training process
        while self.stats["n_tokens"] - self.n_tokens_target < self.stats["last_save_n_tokens"]:
            losses = []
            batch = next(self.loader)
            self.stats["n_steps"] += 1
            # it = self.stats["n_steps"]
            joined_inputs, mask, terminal = batch[0], batch[1], batch[2]

            # get the sub-batch for the current process
            # note: it could be optimized by loading the sub-batch only
            assert len(joined_inputs) == len(mask)
            assert len(joined_inputs) == self.config.load_batch_size 
            assert len(joined_inputs) % self.accelerator.num_processes == 0
            process_batch_size = self.config.load_batch_size  // self.accelerator.num_processes
            i_start = process_batch_size * self.accelerator.process_index
            i_end = process_batch_size * (self.accelerator.process_index + 1)
            joined_inputs, mask = joined_inputs[i_start: i_end], mask[i_start: i_end]

            # decay the learning rate based on our progress
            n_tokens = torch.tensor(np.prod(mask.shape[:-1]), dtype=torch.int64, device=self.accelerator.device)
            n_tokens_sum = self.accelerator.reduce(n_tokens, reduction="sum")
            self.stats["n_tokens"] += n_tokens_sum.item()
 
            overall_progress = self.stats["n_tokens"] / config.final_tokens
            if self.stats["n_tokens"] < config.warmup_tokens:
                # linear warmup
                lr_mult = float(self.stats["n_tokens"]) / float(max(1, config.warmup_tokens))
            else:
                # cosine learning rate decay
                progress = float(self.stats["n_tokens"] - config.warmup_tokens) / float(
                    max(1, config.final_tokens - config.warmup_tokens))
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

            if config.lr_decay:
                lr = config.learning_rate * lr_mult
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = config.learning_rate

            # forward the model
            with self.accelerator.autocast():
                with torch.set_grad_enabled(True):
                    *_, feature, recon_loss, vq_loss, commit_loss,log = model(joined_inputs[:, :-1],
                                                                                            mask[:, :-1], progress=overall_progress)
                    (recon_loss, vq_loss, commit_loss, log) = aggregate_loss([recon_loss, vq_loss, commit_loss, log])
                    loss = (recon_loss+vq_loss+commit_loss).mean()
                    losses.append(loss.item())

            # backprop and update the parameters
            model.zero_grad()

            self.accelerator.backward(loss)
            gradient_norm = self.accelerator.clip_grad_norm_(model.parameters(), config.grad_norm_clip)

            optimizer.step()

            # ema update
            with self.accelerator.autocast():
                gathered_feature = self.accelerator.gather(feature)
                if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
                    model.module.ema_update(torch.flatten(gathered_feature, -1, -1))
                else:
                    model.ema_update(torch.flatten(gathered_feature, -1, -1))

            # report progress
            self.accelerator.wait_for_everyone()
            if self.stats["n_tokens"] - self.stats["last_logging_n_tokens"] > log_freq:
                # CAVEAT: k_tokens_per_second is calculated based on timer in process 0
                time_elapsed = timer() - self.last_logging_time
                tokens_per_sec = (self.stats["n_tokens"]-self.stats["last_logging_n_tokens"]) / time_elapsed
                k_tokens_per_second = tokens_per_sec / 1000
                self.last_logging_time = timer()
                recon_loss = self.accelerator.reduce(recon_loss, reduction="mean")
                state_loss = self.accelerator.reduce(log["state_loss"], reduction="mean")
                action_loss = self.accelerator.reduce(log["action_loss"], reduction="mean")
                reward_loss = self.accelerator.reduce(log["reward_loss"], reduction="mean")
                value_loss = self.accelerator.reduce(log["value_loss"], reduction="mean")
                terminal_loss = self.accelerator.reduce(log["terminal_loss"], reduction="mean")
                commit_loss = self.accelerator.reduce(commit_loss, reduction="mean")
                gradient_norm = self.accelerator.reduce(gradient_norm, reduction="mean")
                summary = dict(recontruction_loss=recon_loss.item(),
                            state_loss=state_loss.item(),
                            action_loss=action_loss.item(),
                            reward_loss=reward_loss.item(),
                            value_loss=value_loss.item(),
                            terminal_loss=terminal_loss.item(),
                            commit_loss=commit_loss.item(),
                            gradient_norm=gradient_norm.item(),
                            lr=lr,
                            lr_mulr=lr_mult,
                            t=time_elapsed
                            )

                # validation
                if dataset.validation_episodes > 0 and self.stats["n_logging"] % validation_period == 0:
                    torch.cuda.empty_cache()
                    model.eval()
                    with self.accelerator.autocast():
                        with torch.set_grad_enabled(False):
                            validation_recon_losses = []
                            validation_commit_losses = []
                            validation_state_losses = []
                            validation_action_losses = []
                            for validation_batch in dataset.validation_set:
                                joined_inputs, mask, terminal = validation_batch[0], validation_batch[1], validation_batch[2]
                                *_, feature, recon_loss, vq_loss, commit_loss, log = model(joined_inputs[:, :-1],
                                                                                        mask[:, :-1],
                                                                                        progress=overall_progress)
                                (recon_loss, vq_loss, commit_loss, log) = aggregate_loss(
                                    [recon_loss, vq_loss, commit_loss, log])
                                validation_recon_losses.append(recon_loss)
                                validation_commit_losses.append(commit_loss)
                                validation_state_losses.append(state_loss)
                                validation_action_losses.append(action_loss)
                            validation_recon_losses = torch.stack(validation_recon_losses).mean()
                            validation_commit_losses = torch.stack(validation_commit_losses).mean()
                            validation_state_losses = torch.stack(validation_state_losses).mean()
                            validation_action_losses = torch.stack(validation_action_losses).mean()
                            validation_recon_losses_gathered = self.accelerator.reduce(validation_recon_losses, reduction="mean")
                            validation_commit_losses_gathered = self.accelerator.reduce(validation_commit_losses, reduction="mean")
                            validation_state_losses_gathered = self.accelerator.reduce(validation_state_losses, reduction="mean")
                            validation_action_losses_gathered = self.accelerator.reduce(validation_action_losses, reduction="mean")
                    summary["validation_recon_loss"] = validation_recon_losses_gathered.item()
                    summary["validation_commit_loss"] = validation_commit_losses_gathered.item()
                    summary["validation_state_loss"] = validation_state_losses_gathered.item()
                    summary["validation_action_loss"] = validation_action_losses_gathered.item()
                    summary["k_tokens_per_second"] = k_tokens_per_second
                    summary["k_tokens"] = self.stats["n_tokens"] / 1000
                    model.train()
                self.accelerator.log(summary, step=self.stats["n_steps"])
                print_string = f'[ utils/training ] epoch {self.stats["n_epochs"]} [ {(self.stats["n_tokens"] - self.stats["last_save_n_tokens"]) // 1000:d}k / 1000k tokens]'
                for k, v in summary.items():
                    print_string += f'{k}: {v:.4f} | '
                self.accelerator.print(print_string)
                self.stats["last_logging_n_tokens"] = self.stats["n_tokens"]
                self.stats["n_logging"] += 1
                    
            # wait for all the processes to finish at the end of each iteration
            self.accelerator.wait_for_everyone()
        
        self.stats["n_epochs"] += 1
        self.stats["last_save_n_tokens"] = self.stats["n_tokens"]

@torch.no_grad()
def get_target_value(joined_inputs, mask, terminal, critic, observation_dim, discount, ignore_terminal=False, prior_model=None):
    observations = joined_inputs[:, :, :observation_dim]

    last = (1-mask[:, :, 0])
    last[:, -1] = 1
    if ignore_terminal:
        terminal = torch.zeros_like(terminal)


    # get the first masked index
    indices = torch.argmax(last, dim=1)
    target_observations = torch.stack([observations[i, indices[i]] for i in range(observations.shape[0])])

    _, q_prime, v_prime, _ = critic(None, target_observations)
    if prior_model == None:
        q_prime = v_prime
    else:
        logits, _, _, _ = prior_model(None, target_observations)
        q_prime = torch.sum(q_prime*torch.softmax(logits, dim=-1), dim=-1)
    q_prime = q_prime[:, 0]

    # get the sum of rewards
    rewards = joined_inputs[:, :-1, -2]
    discount_factors = torch.cat([torch.ones(rewards.shape[0], 1),
                                  torch.cumprod(discount * torch.ones(rewards.shape[0], rewards.shape[1]), dim=1)],
                                 dim=1) * (1-terminal[:, :, 0])
    rewards_sum = torch.sum(rewards*discount_factors[:, :-1]*mask[:, :-1, 0], dim=1).to(q_prime.device)

    q_discount = torch.stack([discount_factors[i, indices[i]] for i in range(discount_factors.shape[0])]).to(q_prime.device)
    target_values = q_prime * q_discount + rewards_sum
    return target_values

@torch.no_grad()
def get_n_step_target_value(joined_inputs, mask, terminal, model, observation_dim, discount, ignore_terminal=False):
    """
    get step specific target values
    Args:
        joined_inputs: [batch_size, seq_len, feature_dim]
        mask: [batch_size, seq_len, 1]
        terminal: [batch_size, seq_len, 1]
        model: model
        observation_dim: observation dimension
        discount: discount factor
        ignore_terminal: ignore terminal signal
    """
    if ignore_terminal:
        terminal = torch.zeros_like(terminal)
    terminal = terminal[:, :, 0]

    # get the first masked index
    observations = joined_inputs[:, :, :observation_dim] # future observations
    B, T, D = observations.shape
    observations = observations.reshape([B*T, D])
    logits, _, q_prime = model(None, observations)
    logits = logits.reshape([B, T, -1])
    q_prime = q_prime.reshape([B, T, -1])

    q_prime = torch.sum(q_prime*torch.softmax(logits, dim=-1), dim=-1)

    unmask = (1-mask[:, :, 0]).to(q_prime.device)
    # get the first masked index
    unmask[:, -1] = 1
    first_unmask_indices = torch.argmax(unmask, dim=1)
    # the q_prime after the first unmasked index should be the same as the q_prime of the first unmasked index
    q_prime_last = torch.stack([q_prime[i, first_unmask_indices[i]] for i in range(q_prime.shape[0])])
    q_prime = q_prime * (1-unmask) + q_prime_last[:,None] * unmask

    rewards = joined_inputs[:, :-1, -2]
    discount_factors = torch.cat([torch.ones(rewards.shape[0], 1),
                                  torch.cumprod(discount * torch.ones(rewards.shape[0], rewards.shape[1]), dim=1)],
                                 dim=1)*(1-terminal)
    cum_rewards = torch.cumsum(rewards*discount_factors[:, :-1]*mask[:, :-1, 0], dim=1).to(q_prime.device)
    discount_factors = discount_factors.to(q_prime.device)

    target_values = q_prime[:, 1:] * discount_factors[:, 1:] + cum_rewards
    return target_values


def compute_gradient_momentum(accelerator, model, targets, states, enable_fp16, max_batch_size=32):
    """
    compute the estimated variance and mean of the gradient
    """

    grads = []
    with accelerator.autocast():
        with torch.set_grad_enabled(True):
            for i in range(min(states.shape[0], max_batch_size)):
                model.zero_grad()
                target = {k: v[i, None] for k, v in targets.items()}
                state = states[i, None]
                _, loss_raw, _, logs = model(target["codes"][:, :-1], state, targets=target)
                accelerator.backward(loss_raw.mean())
                grad_list = []
                for p in model.parameters():
                    if p.grad is not None:
                        grad_list.append(p.grad.view(-1))

                grads.append(torch.cat(grad_list, dim=0))

        grads = torch.stack(grads, dim=0)
        vars = torch.var(grads)
        vars_gathered = accelerator.reduce(vars, reduction="sum")
        means = torch.mean(grads)
        means_gathered = accelerator.reduce(means, reduction="mean")
    return vars_gathered, means_gathered

def get_finetune_weights(target, states, critic_model, overall_progress):
    _, q_values, v_values, logs = critic_model(target["codes"][:, :-1], states, progress=overall_progress, mask=None)
    normed_q_values = critic_model.normalize_returns(q_values)
    weights = torch.clip(normed_q_values, 0, 1)
    return weights

class PriorTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.n_tokens_target = config.n_tokens_target

        self.last_logging_time = timer()
        self.optimizer = None
        self.ema_reconstruction = 0
        self.ema_rate = 0.9

        self.bootstrap_ignore_terminal = config.bootstrap_ignore_terminal

        self.logger = get_logger(__name__, log_level="INFO")

        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=18000))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        mixed_precision = 'fp16' if config.enable_fp16 else None
        self.accelerator = Accelerator(split_batches=True, even_batches=True, log_with="wandb", mixed_precision=mixed_precision, kwargs_handlers=[kwargs, ddp_kwargs])

    def get_optimizer(self, model):
        if self.optimizer is None:
            self.logger.debug(f'[ utils/training ] Making optimizer at epoch {self.stats["n_epochs"]}', main_process_only=True)
            self.optimizer = model.configure_optimizers(self.config)
        return self.optimizer

    def init_stats(self, stats):
        self.stats = stats

    def init_data_loader(self, dataset, stats=None):
        config = self.config

        if not isinstance(dataset, SequentialDataLoader):
            ## seed the dataloader to support interrupted training process
            g = torch.Generator()
            g = g.manual_seed(torch.initial_seed())
            start_idx = 0
            if stats is not None:
                start_idx = stats['num_batch_trained']
            sampler = utils.BatchSampler(dataset, config.load_batch_size, generator=g, start_idx=start_idx)
            loader = DataLoader(
                dataset,
                pin_memory=True,
                batch_sampler=sampler,
                num_workers=config.num_workers,
                worker_init_fn=seed_worker,
                generator=g,
            )
            self.loader = iter(cycle(loader))
        else:
            self.loader = dataset
    
    def init_wandb(self, conf, name):
        self.accelerator.init_trackers("latentPlanning", init_kwargs={"wandb":{"name": name, **conf}})


    def train(self, representation, model, dataset, log_freq=1e5, debug=False, target_policy_model=None, validation_period=12,
              type="prior", prior_model=None, critic_model=None):
        """
        Train the prior or critic model.
        prior_model: the prior model to be used in bootstrapping, only needed for critic training
        """

        config = self.config
        optimizer = self.get_optimizer(model)
        enable_fp16 = config.enable_fp16
        representation.train(False)
        model.train(True)
        nan_loss = False

        while self.stats["n_tokens"] - self.n_tokens_target < self.stats["last_save_n_tokens"]:
            losses = []
            batch = next(self.loader)
            joined_inputs, mask, terminal = batch[0], batch[1], batch[2]
            self.stats["n_steps"] += 1

            # get the sub-batch for the current process
            # note: it could be optimized by loading the sub-batch only
            assert len(joined_inputs) == len(mask) and len(joined_inputs) == len(terminal)
            assert len(joined_inputs) == self.config.load_batch_size 
            assert len(joined_inputs) % self.accelerator.num_processes == 0
            process_batch_size = self.config.load_batch_size  // self.accelerator.num_processes
            i_start = process_batch_size * self.accelerator.process_index
            i_end = process_batch_size * (self.accelerator.process_index + 1)
            joined_inputs, mask, terminal = joined_inputs[i_start: i_end], mask[i_start: i_end], terminal[i_start: i_end]

            n_tokens = torch.tensor(np.prod(joined_inputs[:,:-1].shape[:-1]), dtype=torch.int, device=self.accelerator.device)
            n_tokens_sum = self.accelerator.reduce(n_tokens, reduction="sum")
            self.stats["n_tokens"] += n_tokens_sum.item()
            overall_progress = self.stats["n_tokens"] / config.final_tokens

            if self.stats["n_tokens"] < config.warmup_tokens:
                # linear warmup
                lr_mult = float(self.stats["n_tokens"]) / float(max(1, config.warmup_tokens))
            else:
                # cosine learning rate decay
                progress = float(self.stats["n_tokens"] - config.warmup_tokens) / float(
                    max(1, config.final_tokens - config.warmup_tokens))
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

            if config.lr_decay:
                lr = config.learning_rate * lr_mult
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = config.learning_rate

            if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
                observation_dim = model.module.observation_dim
                action_dim = model.module.action_dim
            else:
                observation_dim = model.observation_dim
                action_dim = model.action_dim
            states = joined_inputs[:, 0, :observation_dim]

            # forward the model
            with self.accelerator.autocast():
                target = {"codes": representation.encode(joined_inputs[:, :-1], mask[:, :-1])}

                if type == "critic":
                    if config.bootstrap:
                        target["value"] = get_target_value(joined_inputs,
                                                        mask, terminal, model,
                                                        observation_dim, config.discount,
                                                        config.bootstrap_ignore_terminal,
                                                        prior_model)
                    else:
                        target["value"] = joined_inputs[:, 0, observation_dim + action_dim + 1]
                    avg_value = torch.mean(target["value"])
                    if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
                        model.module.sync_target_q_heads(0.001)
                    else:
                        model.sync_target_q_heads(0.001)
                elif type == "prior_finetune":
                    target["weights"] = get_finetune_weights(target, states, critic_model, overall_progress)

                with torch.set_grad_enabled(True):
                    if type in "prior":
                        _, loss_raw, _, logs = model(target["codes"][:, :-1], states, targets=target, progress=overall_progress, mask=None)
                    elif type == "critic":
                        loss_raw, _, _, logs = model(target["codes"][:, :-1], states, targets=target, progress=overall_progress, mask=None)
                    elif type == "prior_finetune":
                        _, loss_raw, _, logs = model(target["codes"][:, :-1], states, targets=target, progress=overall_progress, mask=None)
                    (loss, logs) = aggregate_loss([loss_raw, logs])
                    losses.append(loss.mean().item())

            # backprop and update the parameters
            model.zero_grad()

            unscaled_loss = loss.mean()
        
            self.accelerator.backward(unscaled_loss)
            if self.accelerator.sync_gradients:
                gradient_norm = self.accelerator.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            optimizer.step()

            if self.stats["n_tokens"] - self.stats["last_logging_n_tokens"] > log_freq:
                # CAVEAT: k_tokens_per_second is calculated based on the timer in process 0
                time_elapsed = timer() - self.last_logging_time
                tokens_per_sec = (self.stats["n_tokens"]-self.stats["last_logging_n_tokens"]) / time_elapsed
                k_tokens_per_second = tokens_per_sec / 1000        
                self.last_logging_time = timer()

                # aggregate for reporting
                loss = self.accelerator.reduce(unscaled_loss, reduction="mean")
                if torch.isnan(loss).any():
                    nan_loss = True
                    return nan_loss

                gradient_norm = self.accelerator.reduce(gradient_norm, reduction="mean")
                summary = dict(loss=loss.mean().item(),
                        lr=lr,
                        gradient_norm=gradient_norm.item(),
                        lr_mulr=lr_mult, )
                for k, v in logs.items():
                    summary[k] = v.mean().item()
                if type == "critic":
                    summary["avg_value"] = avg_value.item()

                # validation
                if dataset.validation_episodes > 0 and self.stats["n_logging"] % validation_period == 0:
                    if torch.cuda.get_device_properties(0).total_memory == 25444024320 and False: # 24GB RTX3090/4090
                        # Note that compute_gradient_momentum() is blocking as it gathers grads from all processes
                        gradient_variances, gradient_mean = compute_gradient_momentum(self.accelerator, model, targets=target, states=states,
                                                                                    enable_fp16=enable_fp16)
                        summary['gradient_noise_scale'] = gradient_variances.sum()/torch.norm(gradient_mean)

                    torch.cuda.empty_cache()
                    model.eval()
                    with self.accelerator.autocast():
                        with torch.set_grad_enabled(False):
                            validation_losses = []
                            for validation_batch in dataset.validation_set:
                                joined_inputs, mask, terminal = validation_batch
                                states = joined_inputs[:, 0, :observation_dim]
                                target = {"codes": representation.encode(joined_inputs[:, :-1], mask[:, :-1])}
                                if type == "critic":
                                    if config.bootstrap:
                                        target["value"] = get_target_value(joined_inputs,
                                                                           mask, terminal, model,
                                                                           observation_dim, config.discount,
                                                                           config.bootstrap_ignore_terminal,
                                                                           prior_model)
                                    else:
                                        target["value"] = joined_inputs[:, 0, observation_dim + action_dim + 1]
                                elif type == "prior_finetune":
                                    target["weights"] = get_finetune_weights(target, states, critic_model, overall_progress)

                                if type in ["prior", "prior_finetune"]:
                                    _, validation_loss, _, logs = model(target["codes"][:, :-1], states, targets=target, mask=None)
                                elif type == "critic":
                                    validation_loss, _, _, logs = model(target["codes"][:, :-1], states, targets=target, mask=None)
                                (validation_loss, logs) = aggregate_loss([validation_loss, logs])
                                validation_losses.append(validation_loss.mean())
                            # gather losses from all processes for metric report
                            validation_losses = torch.stack(validation_losses, dim=0).mean()
                            validation_losses_gathered = self.accelerator.reduce(validation_losses, reduction="mean")
                            summary["validation_loss"] = validation_losses_gathered.mean().item()
                    model.train()

                summary["k_tokens_per_second"] = k_tokens_per_second
                summary["k_tokens"] = self.stats["n_tokens"] / 1000
                self.accelerator.log(summary, step=self.stats["n_steps"])
                print_string = f'[ utils/training ] epoch {self.stats["n_epochs"]} [ {(self.stats["n_tokens"] - self.stats["last_save_n_tokens"]) // 1000:d}k / 1000k tokens]'
                for k, v in summary.items():
                    print_string += f'{k}: {v:.4f} | '
                self.accelerator.print(print_string)

                self.stats["last_logging_n_tokens"] = self.stats["n_tokens"]
                self.stats["n_logging"] += 1
                self.ema_reconstruction = self.ema_reconstruction * self.ema_rate + loss.mean().item() * (1 - self.ema_rate)

            # wait for all the processes to finish at the end of each iteration
            self.accelerator.wait_for_everyone()

        self.stats["n_epochs"] += 1
        self.stats["last_save_n_tokens"] = self.stats["n_tokens"]
        return nan_loss
