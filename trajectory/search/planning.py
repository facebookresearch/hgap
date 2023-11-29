from collections import defaultdict
import torch
from trajectory.utils.relabel_humanoid import get_speed, get_angular_vel, get_forward_vel, get_height_vel, get_left_vel


REWARD_DIM = VALUE_DIM = 1

import numpy as np


def trajectory2rv(trajectories, objective):
    """
        trajectories: [B x T x D]
    """
    if objective == "reward":
        return trajectories[:, :, -3], trajectories[:, :, -2]
    elif objective == "speed":
        speed = get_speed(trajectories)
        return speed, speed
    elif objective == "shift_left":
        left_vel = get_left_vel(trajectories)
        return left_vel, left_vel
    elif objective == "forward":
        forward_vel = get_forward_vel(trajectories)
        return forward_vel, forward_vel
    elif objective == "backward":
        forward_vel = get_forward_vel(trajectories)
        return -forward_vel, -forward_vel
    elif objective == "jump":
        height_vel = get_height_vel(trajectories)
        jump_vel = torch.maximum(height_vel, torch.zeros_like(height_vel))
        return jump_vel, jump_vel
    elif objective == "rotate_x":
        x_angular_vel = get_angular_vel(trajectories, "x")
        return x_angular_vel, x_angular_vel
    elif objective == "rotate_y":
        y_angular_vel = get_angular_vel(trajectories, "y")
        return y_angular_vel, y_angular_vel
    elif objective == "rotate_z":
        z_angular_vel = get_angular_vel(trajectories, "z")
        return z_angular_vel, z_angular_vel
    elif objective == "tracking":
        zeros = torch.zeros([trajectories.shape[0], trajectories.shape[1]]).to(trajectories)
        return zeros, zeros


@torch.no_grad()
def sample_with_prior(prior, model, x, denormalize_rew, denormalize_val, discount, steps, nb_samples=4096, rounds=8,
                      likelihood_weight=5e2, prob_threshold=0.05, uniform=False, return_info=False, objective="reward",
                      temperature=1.0, top_p=0.95):
    x = x.to(torch.float32)
    state = x[:, 0, :model.observation_dim]
    optimals = []
    optimal_values = []
    log_prob_list = []
    entropy_list = []
    info = defaultdict(list)
    for round in range(rounds):
        contex = None
        samples = None
        log_acc_probs = torch.zeros([1]).to(x)
        kv_cache = None
        for step in range(steps // model.latent_step):
            for internal_step in range(model.code_per_step):
                logits, _, kv_cache, _ = prior(samples.reshape([-1, 1]) if samples is not None else None,
                                               state, kv_cache=kv_cache)  # [B x t x K]
                logits = logits / temperature  # Add temperature control
                probs = torch.softmax(logits[:, -1, :], dim=-1)  # [B x K]
                entropy_list.append(torch.mean(-torch.sum(probs * torch.log(probs + 1e-8), dim=-1)))
                log_probs = torch.log(probs)

                # Top-p sampling
                sorted_probs, indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs < top_p
                mask[:, 0] = 1  # To make sure the most probable token is always included in the sampling set
                valid_probs = sorted_probs*mask
                if step == 0 and internal_step == 0:
                    sampled_indices = torch.multinomial(valid_probs, num_samples=nb_samples // rounds,
                                                        replacement=True)
                else:
                    sampled_indices = torch.multinomial(valid_probs.reshape(nb_samples, -1),
                                                        num_samples=1, replacement=True)
                samples = torch.gather(indices, 1, sampled_indices)

                samples_log_prob = torch.cat(
                    [torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(log_probs, samples)])  # [B, M]

                log_prob_list.append(samples_log_prob.squeeze())
                log_acc_probs = log_acc_probs + samples_log_prob.reshape([-1])
                if contex is not None:
                    contex = torch.cat([contex, samples.reshape([-1, 1])], dim=1)
                else:
                    contex = samples.reshape([-1, step * model.code_per_step + internal_step + 1])  # [(B*M) x t]

        prediction_raw = model.decode_from_indices(contex, state)

        r, v = trajectory2rv(prediction_raw, objective=objective)

        discounts = torch.cumprod(torch.ones_like(r) * discount, dim=-1)
        values = torch.sum(r[:, :-1] * discounts[:, :-1], dim=-1) + v[:, -1] * discounts[:, -1]


        likelihood_bonus = likelihood_weight * torch.minimum(log_acc_probs, torch.log(torch.tensor(prob_threshold)))
        info["log_probs"].append(log_acc_probs.cpu().numpy())
        info["log_prob_list"].append([prob.cpu().numpy().squeeze() for prob in log_prob_list])
        info["returns"].append(values.cpu().numpy())
        info["predictions"].append(prediction_raw.cpu().numpy())
        info["objectives"].append(values.cpu().numpy() + likelihood_bonus.cpu().numpy())
        info["latent_codes"].append(contex.cpu().numpy())
        info["entropy"].append(torch.stack(entropy_list).cpu().numpy())
        max_idx = (values + likelihood_bonus).argmax()
        optimal_value = values[max_idx]
        optimal = prediction_raw[max_idx]
        optimals.append(optimal)
        optimal_values.append(optimal_value.item())

    for key, val in info.items():
        info[key] = np.concatenate(val, axis=0)

    max_idx = np.array(optimal_values).argmax()
    optimal = optimals[max_idx]
    # for key in ["returns", "objectives"]:
    #     val = info[key]
    #     print(f"{key} {val},\n with mean: {np.mean(val)}, std {np.std(val)} \n")

    if return_info:
        return optimal.cpu().numpy(), info
    else:
        return optimal.cpu().numpy()


@torch.no_grad()
def beam_with_prior(prior, model, x, denormalize_rew, denormalize_val, discount, steps,
                    beam_width, n_expand, prob_threshold=0.05, likelihood_weight=5e2, prob_acc="product", return_info=False,
                    optimize_target="reward", temperature=1.0, top_p=0.99):
    contex = None
    state = x[:, 0, :prior.observation_dim]
    acc_probs = torch.zeros([1]).to(x)
    info = {}
    kv_cache = None
    for step in range(steps//model.latent_step):
        for internal_step in range(model.code_per_step):
            logits, _, new_kv_cache, _ = prior(contex, state) # [B x t x K]
            logits = logits / temperature  # Add temperature control
            probs = torch.softmax(logits[:, -1, :], dim=-1) # [B x K]
            log_probs = torch.log(probs)
            nb_samples = beam_width * n_expand if step == 0 and internal_step == 0 else n_expand
            samples = torch.multinomial(probs, num_samples=nb_samples, replacement=True) # [B, M]
            samples_log_prob = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(log_probs, samples)]) # [B, M]
            if prob_acc in ["product", "expect"]:
                acc_probs = acc_probs.repeat_interleave(nb_samples, 0) + samples_log_prob.reshape([-1])
            elif prob_acc == "min":
                acc_probs = torch.minimum(acc_probs.repeat_interleave(nb_samples, 0), samples_log_prob.reshape([-1]))

            if not contex is None:
                contex = torch.cat([torch.repeat_interleave(contex, nb_samples, 0), samples.reshape([-1, 1])],
                                   dim=1)
            else:
                contex = samples.reshape([-1, step+1]) # [(B*M) x t]


            if internal_step==model.code_per_step-1:
                prediction_raw = model.decode_from_indices(contex, state)
                prediction = prediction_raw.reshape([-1, prediction_raw.shape[-1]])

                r_t, V_t = trajectory2rv(prediction, optimize_target)
                if denormalize_rew is not None:
                    r_t = denormalize_rew(r_t).reshape([contex.shape[0], -1])
                if denormalize_val is not None:
                    V_t = denormalize_val(V_t).reshape([contex.shape[0], -1])
                if return_info:
                    info[(step + 1) * model.latent_step] = dict(predictions=prediction_raw.cpu(), returns=values.cpu(),
                                                                latent_codes=contex.cpu(), log_probs=acc_probs.cpu(),
                                                                objectives=values + likelihood_bonus, index=index.cpu())
                discounts = torch.cumprod(torch.ones_like(r_t) * discount, dim=-1)
                values = torch.sum(r_t[:, :-1] * discounts[:, :-1], dim=-1) + V_t[:, -1] * discounts[:, -1]
                if prob_acc == "product":
                    likelihood_bonus = likelihood_weight * torch.clamp(acc_probs, 0, np.log(prob_threshold) * (
                                steps // model.latent_step))
                elif prob_acc == "min":
                    likelihood_bonus = likelihood_weight * torch.clamp(acc_probs, 0, np.log(prob_threshold))
            else:
                values = torch.zeros([contex.shape[0]]).to(x)
                likelihood_bonus = acc_probs

            nb_top = 1 if step == (steps//model.latent_step-1) and internal_step == (model.code_per_step-1) else beam_width
            if prob_acc == "expect":
                values_with_b, index = torch.topk(values*torch.exp(acc_probs), nb_top)
            else:
                values_with_b, index = torch.topk(values+likelihood_bonus, nb_top)

            contex = contex[index]
            acc_probs = acc_probs[index]

    optimal = prediction_raw[index[0]]
    if return_info:
        return optimal.cpu().numpy(), info
    else:
        return optimal.cpu().numpy()

@torch.no_grad()
def top_k(prior, model, x, denormalize_rew, denormalize_val, discount, steps,
          k, nb_samples, prob_threshold=0.05, likelihood_weight=5e2, return_info=False, optimize_target="reward"):
    x = x.to(torch.float32)
    contex = None
    state = x[:, 0, :prior.observation_dim]
    info = {}
    log_prob_list = []
    log_acc_probs = torch.zeros([1]).to(x)
    for step in range(steps//model.latent_step):
        for internal_step in range(model.code_per_step):
            logits, _, values = prior(contex, state) # [B x t x M]
            logits = logits[:, -1, :]
            logits, candidate_latent_codes = torch.topk(logits, k, dim=-1)

            probs = torch.softmax(logits[:, :], dim=-1) # [B x K]
            log_probs = torch.log(probs)

            if step == 0 and internal_step == 0:
                samples_idx = torch.multinomial(probs, num_samples=nb_samples, replacement=True)  # [B, K]
            else:
                samples_idx = torch.multinomial(probs, num_samples=1, replacement=True)  # [B, K]
            latent_code = torch.cat([torch.index_select(c, 0, i) for c,i in zip(candidate_latent_codes, samples_idx)])
            samples_log_prob = torch.cat(
                [torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(log_probs, samples_idx)])  # [B, K]
            log_prob_list.append(samples_log_prob.squeeze())
            log_acc_probs = log_acc_probs + samples_log_prob.reshape([-1])
            if not contex is None:
                contex = torch.cat([contex, latent_code.reshape([-1, 1])], dim=1)
            else:
                contex = latent_code.reshape([-1, step * model.code_per_step + internal_step + 1])  # [(B*M) x t]

    prediction_raw = model.decode_from_indices(contex, state)
    prediction = prediction_raw.reshape([-1, model.action_dim+model.observation_dim+3])
    likelihood_bonus = likelihood_weight * torch.min(torch.stack(log_prob_list, dim=-1),
                                                     torch.log(torch.tensor(prob_threshold)))
    likelihood_bonus = torch.sum(likelihood_bonus, dim=-1)

    r_t, V_t = prediction2rv(prediction, model, optimize_target)

    if denormalize_rew is not None:
        r_t = denormalize_rew(r_t).reshape([contex.shape[0], -1])
    if denormalize_val is not None:
        V_t = denormalize_val(V_t).reshape([contex.shape[0], -1])

    discounts = torch.cumprod(torch.ones_like(r_t) * discount, dim=-1)
    values = torch.sum(r_t[:,:-1] * discounts[:, :-1], dim=-1) + V_t[:,-1]*discounts[:, -1]
    #likelihood_bonus = likelihood_weight*torch.clamp(log_probs, -1e5, np.log(prob_threshold)*(steps//model.latent_step))
    values_with_b, index = torch.topk(values+likelihood_bonus, 1)
    if return_info:
        info = dict(predictions=prediction_raw.cpu(), returns=values.cpu(), latent_codes=contex.cpu(),
                    log_probs=torch.stack(log_prob_list).cpu().sum(),
                    log_prob_list=[prob.cpu().numpy().squeeze() for prob in log_prob_list],
                    objectives=values+likelihood_bonus, index=index.cpu())

    optimal = prediction_raw[index[0]]
    if return_info:
        return optimal.cpu().numpy(), info
    else:
        return optimal.cpu().numpy()
    
