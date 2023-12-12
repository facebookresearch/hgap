# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Function
from trajectory.models.transformers import *
from trajectory.models.cnn import ResidualTemporalBlock, ResidualTemporalDeConvBlock
from trajectory.models.ein import EinLinear
from rotary_embedding_torch import RotaryEmbedding
import numpy as np
from trajectory.utils.symlog import symlog, symexp
from collections import namedtuple

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            output_size = inputs_size[:-1] + (inputs_size[-1]//embedding_size,)
            inputs_flatten = inputs.reshape(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*output_size)
            ctx.mark_non_differentiable(indices)
            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply

class VQEmbeddingMovingAverage(nn.Module):
    def __init__(self, K, D, slice=1, decay=0.99):
        super().__init__()
        embedding = torch.zeros(K, D//slice)
        embedding.uniform_(-1./K, 1./K)
        self.decay = decay
        self.slice = slice

        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.ones(K))
        self.register_buffer("ema_w", self.embedding.clone())

    def straight_through(self, z_e_x, data_parallel=False):
        K, D = self.embedding.size()

        z_e_x_ = z_e_x.contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding)
        z_q_x = z_q_x_.contiguous()

        if self.training and not data_parallel:
            #ema_w_slice = self.ema_w[:, slice_start:slice_end].clone()
            encodings = F.one_hot(indices, K).float()
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            dw = encodings.transpose(1, 0)@z_e_x_.reshape([-1, D])
            self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw

            self.embedding = self.ema_w / (self.ema_count.unsqueeze(-1))
            self.embedding = self.embedding.detach()
            self.ema_count = self.ema_count.detach()
            self.ema_w = self.ema_w.detach()

        z_q_x_bar_flatten = torch.index_select(self.embedding, dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.contiguous()

        return z_q_x, z_q_x_bar
    
    def ema_update(self, z_e_x):
        K, D = self.embedding.size()

        z_e_x_ = z_e_x.contiguous()
        _, indices = vq_st(z_e_x_, self.embedding)

        encodings = F.one_hot(indices, K).float()
        self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

        dw = encodings.transpose(1, 0)@z_e_x_.reshape([-1, D])
        self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw

        self.embedding = self.ema_w / (self.ema_count.unsqueeze(-1))
        self.embedding = self.embedding.detach()
        self.ema_count = self.ema_count.detach()
        self.ema_w = self.ema_w.detach()


class VQEmbedding(nn.Module):
    def __init__(self, K, D, slice=1):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)
        self.slice = slice

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.contiguous()
        latents = []
        for i in range(self.slice):
            latents.append(vq(z_e_x_, self.embedding.weight[:, i*self.embedding.size(1)//self.slice:(i+1)*self.embedding.size(1)//self.slice]))
        return torch.cat(latents, dim=-1)

    def straight_through(self, z_e_x, data_parallel=False):
        z_q_x_list = []
        z_q_x_bar_list = []
        for i in range(self.slice):
            slice_start = i*self.embedding.size(1)//self.slice
            slice_end = (i+1)*self.embedding.size(1)//self.slice
            z_e_x_ = z_e_x.contiguous()[..., slice_start:slice_end]
            z_q_x_, indices = vq_st(z_e_x_, self.embedding[:, slice_start:slice_end])
            z_q_x = z_q_x_.contiguous()

            z_q_x_bar_flatten = torch.index_select(self.embedding, dim=0, index=indices)
            z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
            z_q_x_bar = z_q_x_bar_.contiguous()
            z_q_x_list.append(z_q_x)
            z_q_x_bar_list.append(z_q_x_bar)
        return torch.cat(z_q_x_list, dim=-1), torch.cat(z_q_x_bar_list, dim=-1)

class VQAutoencoder(nn.Module):
    def __init__(self, config, feature_dim):
        super().__init__()
        if "data_parallel" in config:
            self.data_parallel = config.data_parallel
        else:
            self.data_parallel = False 
        self.K=config.K
        self.latent_size = config.trajectory_embd
        self.condition_size = config.observation_dim
        self.trajectory_input_length = config.block_size - config.transition_dim
        self.embedding_dim = config.n_embd
        self.trajectory_length = config.block_size//config.transition_dim-1
        self.block_size = config.block_size
        self.observation_dim = feature_dim
        self.action_dim = config.action_dim
        self.encoder_inputs = config.encoder_inputs
        self.transition_dim = config.transition_dim
        self.symlog = config.symlog

        if ("reward" not in self.encoder_inputs) and "return" not in self.encoder_inputs:
            self.no_reward_value = True
        else:
            self.no_reward_value = False

        if self.no_reward_value:
            self.transition_dim -= 2
        self.latent_step = config.latent_step
        self.state_conditional = config.state_conditional
        if "code_per_step" not in config:
            self.code_per_step = 1
        else:
            self.code_per_step = config.code_per_step
        if "position_encoding" in config:
            self.pos_embd_type = config.position_embedding
        else:
            self.pos_embd_type = "absolute"
        causal_conv = True if "causal_conv" not in config else config.causal_conv


        self.input_masks = torch.zeros(1, 1, self.transition_dim)
        if "state" in self.encoder_inputs:
            self.input_masks[0, 0, :self.observation_dim] = 1
        if "action" in self.encoder_inputs:
            self.input_masks[0, 0, self.observation_dim:self.observation_dim+self.action_dim] = 1

        if not self.no_reward_value:
            if "reward" in self.encoder_inputs:
                self.input_masks[0, 0, -3] = 1
            if "return" in self.encoder_inputs:
                self.input_masks[0, 0, -2] = 1

        if "mask" in self.encoder_inputs or "terminal" in self.encoder_inputs:
            self.input_masks[0, 0, -1] = 1

        self.masking = "none"

        out_feature_size = self.embedding_dim

        self.pos_emb = nn.Parameter(torch.zeros(1, self.trajectory_length, config.n_embd))
        self.embed = nn.Linear(self.transition_dim, self.embedding_dim)
        self.drop = nn.Dropout(config.embd_pdrop)

        if config.ae_type == "Transformer":
            self.encoder = nn.Sequential(*[Block(config.n_embd, config.n_head, config.resid_pdrop,
                                            config.attn_pdrop, sequence_length=self.trajectory_length,
                                            causal=config.causal_attention, rotary=(config.position_embedding=="rotary")) for _ in range(config.n_layer)])
        elif config.ae_type == "CNN":
            self.encoder = nn.Sequential(*[ResidualTemporalBlock(self.embedding_dim, self.embedding_dim) for _ in range(config.n_layer)])
        elif config.ae_type == "StrideCNN":
            layers = []
            in_feature_size = self.embedding_dim
            for _ in range(config.n_layer-1):
                out_feature_size = in_feature_size*2
                for _ in range(config.blocks_per_layer):
                    layers.append(ResidualTemporalBlock(in_feature_size, in_feature_size))
                layers.append(ResidualTemporalBlock(in_feature_size, out_feature_size, stride=2))
                in_feature_size = out_feature_size
            non_pooling_layer = [ResidualTemporalBlock(in_feature_size, out_feature_size)]
            self.encoder = nn.Sequential(*layers+non_pooling_layer)
        elif config.ae_type == "AttentionCNN":
            attention_layer = [Block(config.n_embd, config.n_head, config.resid_pdrop,
                                            config.attn_pdrop, sequence_length=self.trajectory_length,
                                            causal=config.causal_attention, rotary=(config.position_embedding=="rotary"))]
            self.encoder = nn.Sequential(*[ResidualTemporalBlock(self.embedding_dim, self.embedding_dim, causal=causal_conv)
                                           for _ in range(config.n_layer-1)]+attention_layer)
        
        if "ma_update" in config and not (config.ma_update):
            self.codebook = VQEmbedding(config.K, config.trajectory_embd, self.code_per_step)
            self.ma_update = False
        else:
            self.codebook = VQEmbeddingMovingAverage(config.K, config.trajectory_embd, self.code_per_step)
            self.ma_update = True

        if config.ae_type != "StrideCNN":
            self.latent_mixing = nn.Linear(self.latent_size+self.observation_dim, self.embedding_dim)
        else:
            self.latent_mixing = nn.Linear(self.latent_size+self.observation_dim, out_feature_size)

        if config.ae_type == "Transformer":
            self.decoder = nn.Sequential(*[Block(config.n_embd, config.n_head, config.resid_pdrop,
                                            config.attn_pdrop, sequence_length=self.trajectory_length,
                                            causal=config.causal_attention, rotary=(config.position_embedding=="rotary")) for _ in range(config.n_layer)])
        elif config.ae_type == "CNN":
            self.decoder = nn.Sequential(*[ResidualTemporalBlock(self.embedding_dim, self.embedding_dim) for _ in range(config.n_layer)])
        elif config.ae_type == "StrideCNN":
            layers = []
            non_pooling_layer = [ResidualTemporalBlock(in_feature_size, out_feature_size)]
            for _ in range(config.n_layer - 1):
                out_feature_size = in_feature_size // 2
                layers.append(ResidualTemporalDeConvBlock(in_feature_size, out_feature_size, stride=2))
                for _ in range(config.blocks_per_layer):
                    layers.append(ResidualTemporalBlock(out_feature_size, out_feature_size))
                in_feature_size = out_feature_size
            self.decoder = nn.Sequential(*non_pooling_layer+layers)
        elif config.ae_type == "AttentionCNN":
            attention_layer = [Block(config.n_embd, config.n_head, config.resid_pdrop,
                                    config.attn_pdrop, sequence_length=self.trajectory_length,
                                    causal=config.causal_attention, rotary=(config.position_embedding=="rotary"))]
            self.decoder = nn.Sequential(*attention_layer+[ResidualTemporalBlock(self.embedding_dim, self.embedding_dim, causal=causal_conv)
                                                           for _ in range(config.n_layer-1)])

        self.ae_type = config.ae_type

        self.embed = nn.Linear(self.transition_dim, self.embedding_dim)
        self.predict = nn.Linear(self.embedding_dim, self.transition_dim)
        self.cast_embed = nn.Linear(out_feature_size, self.latent_size)

        self.bottleneck = config.bottleneck

        if self.bottleneck == "pooling":
            self.latent_pooling = nn.MaxPool1d(self.latent_step, stride=self.latent_step)
        else:
            raise ValueError(f'Unknown bottleneck type {self.bottleneck}')

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.predict = nn.Linear(self.embedding_dim, self.transition_dim)
        self.residual = config.residual


    def encode(self, joined_inputs):
        if self.no_reward_value:
            joined_inputs = torch.cat([joined_inputs[:, :, :-3], joined_inputs[:, :, -1, None]], dim=-1)
        joined_inputs = joined_inputs.to(self.embed.weight)
        b, t, joined_dimension = joined_inputs.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        self.input_masks = self.input_masks.to(joined_inputs.device)
        joined_inputs = self.input_masks * joined_inputs
        # forward the GPT model
        token_embeddings = self.embed(joined_inputs)

        if self.ae_type in ["Transformer", "AttentionCNN"]:
            ## [ 1 x T x embedding_dim ]
            if "absolute" in self.pos_embd_type:
                position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
                ## [ B x T x embedding_dim ]
                token_embeddings = token_embeddings + position_embeddings

        x = self.drop(token_embeddings)
        x = self.encoder(x)
        ## [ B x T x embedding_dim ]

        x = self.cast_embed(x)
        ## [ B x T x trajectory_embd ]

        if self.ae_type == "StrideCNN":
            pass
        elif self.bottleneck == "pooling":
            x = self.latent_pooling(x.transpose(1, 2)).transpose(1, 2)
        elif self.bottleneck == "attention":
            x = self.latent_pooling(x)
        else:
            raise ValueError()
        return x

    def decode(self, latents, state):
        """
            latents: [B x (T//self.latent_step*self.code_per_step) x latent_size]
            state: [B x observation_dimension]
        """
        B, T, _ = latents.shape

        if self.ae_type == "StrideCNN":
            pass
        elif self.bottleneck == "pooling":
            latents = torch.repeat_interleave(latents, self.latent_step, dim=1)
        elif self.bottleneck == "attention":
            latents = self.expand(latents)

        state_flat = torch.reshape(state, shape=[B, 1, -1]).repeat(1, T*self.latent_step, 1)
        if self.symlog:
            state_flat = symlog(state_flat)

        if not self.state_conditional:
            state_flat = torch.zeros_like(state_flat)

        inputs = torch.cat([state_flat, latents], dim=-1)
        inputs = self.latent_mixing(inputs)

        if self.ae_type in ["Transformer", "AttentionCNN"] and "absolute" in self.pos_embd_type:
            inputs = inputs + self.pos_emb.to(inputs.device)[:, :inputs.shape[1]]

        x = self.decoder(inputs)
        x = self.ln_f(x)

        ## [B x T x obs_dim]
        joined_pred = self.predict(x)
        if self.residual == "absolute":
            joined_pred[:, 0, :self.observation_dim] = 0  # force the first state to be the same as the input state
        elif self.residual == "relative":
            joined_pred[:, 0, :self.observation_dim] = 0  # force the first state to be the same as the input state
            joined_pred[:, :, :self.observation_dim] = torch.cumsum(joined_pred[:, :, :self.observation_dim], dim=1)
        if self.symlog:
            joined_pred[:, :, :self.observation_dim] += symlog(torch.reshape(state, shape=[B, 1, -1]).to(joined_pred.device))
        else:
            joined_pred[:, :, :self.observation_dim] += torch.reshape(state, shape=[B, 1, -1]).to(joined_pred.device)
        return joined_pred

    def forward(self, joined_inputs, state):
        trajectory_feature = self.encode(joined_inputs)
        latents_st, latents = self.codebook.straight_through(trajectory_feature, data_parallel=self.data_parallel)
        trajectory_feature_masked = trajectory_feature.clone()
        joined_pred = self.decode(latents_st, state.to(latents_st.device))
        return joined_pred, latents.to(joined_pred.device), trajectory_feature.to(joined_pred.device), trajectory_feature_masked.to(joined_pred.device)


class VQContinuousVAE(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        if "data_parallel" in config:
            self.data_parallel = config.data_parallel
        else:
            self.data_parallel = False 
        # input embedding stem (+1 for stop token)
        self.model = VQAutoencoder(config, config.observation_dim)
        self.trajectory_embd = config.trajectory_embd
        self.K = config.K
        self.vocab_size = config.vocab_size
        self.stop_token = config.vocab_size * config.transition_dim
        self.block_size = config.block_size
        self.observation_dim = config.observation_dim
        self.masking = config.masking
        self.code_per_step = config.code_per_step

        self.action_dim = config.action_dim
        self.trajectory_length = config.block_size//config.transition_dim-1
        self.transition_dim = config.transition_dim
        self.action_weight = config.action_weight
        self.reward_weight = config.reward_weight # * float('reward' in config.encoder_inputs)
        self.value_weight = config.value_weight # * float('value' in config.encoder_inputs)
        self.position_weight = config.position_weight
        self.latent_step = config.latent_step
        self.use_discriminator = config.use_discriminator
        self.padding_vector = torch.zeros(self.transition_dim-1)
        self.apply(self._init_weights)
        self.ema_reconstruction = 0
        self.ema_rate = 0.9
        self.symlog = config.symlog

    def get_last_layer(self):
        return self.model.predict.weight

    def get_block_size(self):
        return self.block_size

    def set_padding_vector(self, padding):
        self.padding_vector = padding

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, EinLinear, nn.Conv1d, nn.ConvTranspose1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, nn.GroupNorm, nn.BatchNorm1d)
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('norm.weight'):
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        if isinstance(self.model, VQAutoencoder):
            if "absolute" in self.model.pos_embd_type:
                no_decay.add('pos_emb')
            if self.model.bottleneck == "attention":
                no_decay.add('latent_pooling.query')
                no_decay.add('expand.query')
                no_decay.add('latent_pooling.attention.in_proj_weight')
                no_decay.add('expand.attention.in_proj_weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)

        if not self.use_discriminator:
            return optimizer
        else:
            disc_optimizer = torch.optim.AdamW(self.discriminator_loss.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
            return optimizer, disc_optimizer

    @torch.no_grad()
    def encode(self, joined_inputs, masks):
        b, t, joined_dimension = joined_inputs.size()
        padded = torch.tensor(self.padding_vector, dtype=torch.float32,
                              device=joined_inputs.device).repeat(b, t, 1)
        joined_inputs = joined_inputs*masks+(1-masks)*padded

        trajectory_feature = self.model.encode(torch.cat([joined_inputs, masks], dim=2))
        if self.model.ma_update:
            indices = vq(trajectory_feature, self.model.codebook.embedding)
        else:
            indices = vq(trajectory_feature, self.model.codebook.embedding.weight)
        return indices.reshape(b, -1)

    def decode(self, latent, state):
        """
        Decode a trajectory from feature vectors.
        Args:
            latent: [B x T x D] latent feature vectors
            state: [B x obs_dim] initial states
        """
        joined_pred = self.model.decode(latent, state)
        if self.symlog:
            joined_pred[:, :, :-1] = symexp(joined_pred[:, :, :-1])
        joined_pred[:, :, -1] = torch.sigmoid(joined_pred[:, :, -1])
        return joined_pred

    def decode_from_indices(self, indices, state):
        """
        Decode a trajectory from latent codes
        Args:
            indices: [B x T] latent codes
            state: [B x obs_dim] or [1 x obs_dim] initial state. [1 x obs_dim] is a shortcut for planning
        """
        B, T = indices.shape
        if self.model.ma_update:
            latent = torch.index_select(self.model.codebook.embedding, dim=0, index=indices.flatten()).reshape([B, T, -1])
        else:
            latent = torch.index_select(self.model.codebook.embedding.weight, dim=0, index=indices.flatten()).reshape(
                [B, T, -1])
        state = state[:,None,:]
        if state.shape[0] == 1 and B > 1:
            state = state.repeat(B, 1, 1)
        joined_pred = self.decode(latent.reshape([B, T//self.code_per_step, -1]), state)
        return joined_pred

    def forward(self, joined_inputs, mask, progress=None):
        """
        Run the full autoencoder model on the given inputs and get loss.
        Args:
            joined_inputs : [ B x T x joined_dimension]
            mask : [ B x T x 1]
            progress: used for discriminator training (optional)
        """

        joined_inputs = joined_inputs.to(dtype=torch.float32, device=self.model.embed.weight.device)
        b, t, joined_dimension = joined_inputs.size()
        padded = torch.tensor(self.padding_vector, dtype=torch.float32,
                              device=joined_inputs.device).repeat(b, t, 1)

        raw_mask = mask.to(joined_inputs.device)
        if mask.shape[-1] == 1:
            mask = torch.clone(raw_mask).repeat(1, 1, joined_inputs.shape[-1])
        joined_inputs = joined_inputs*mask+(1-mask)*padded

        state = joined_inputs[:,0,:self.observation_dim]
        
        # with model.join():
        ## [ B x T x embedding_dim ]
        # forward the GPT model
        reconstructed_logits, latents, feature, feature_masked = self.model(torch.cat([joined_inputs, raw_mask], dim=2),
                                                                     state)

        pred_trajectory = torch.reshape(reconstructed_logits[:, :, :-1], shape=[b, t, self.model.transition_dim-1])
        pred_mask = reconstructed_logits[:, :, -1, None]
        if self.symlog:
            reconstructed = torch.cat([symexp(pred_trajectory), torch.sigmoid(pred_mask)], dim=2)
        else:
            reconstructed = torch.cat([pred_trajectory, torch.sigmoid(pred_mask)], dim=2)

        # if we are given some desired targets also calculate the loss
        logs = {}
        if self.model.ma_update:
            loss_vq = torch.tensor(0.0, device=joined_inputs.device)
        else:
            loss_vq = F.mse_loss(latents, feature_masked.detach())
        # Commitment objective
        loss_commit = F.mse_loss(feature_masked, latents.detach())

        weights = torch.cat([
            torch.ones(2, device=joined_inputs.device)*self.position_weight,
            torch.ones(self.observation_dim-2, device=joined_inputs.device),
            torch.ones(self.action_dim, device=joined_inputs.device) * self.action_weight,
            torch.ones(1, device=joined_inputs.device) * self.reward_weight,
            torch.ones(1, device=joined_inputs.device) * self.value_weight,
        ])


        if self.model.no_reward_value:
            weights = weights[:-2]
            joined_inputs = joined_inputs[:,:,:-2]
            mask = mask[:,:,:-2]
        if self.symlog:
            mse = F.mse_loss(pred_trajectory, symlog(joined_inputs), reduction='none') * weights[None, None, :]
        else:
            mse = F.mse_loss(pred_trajectory, joined_inputs, reduction='none') * weights[None, None, :]
        mse = mse*mask

        cross_entropy = F.binary_cross_entropy_with_logits(pred_mask, torch.clip(raw_mask.float(), 0.0, 1.0))
        reconstruction_loss = mse.mean()+cross_entropy

        logs['action_loss'] = torch.mean(mse[:, :, self.observation_dim:self.observation_dim+self.action_dim].detach())
        logs['state_loss'] = torch.mean(mse[:, :, :self.observation_dim].detach())

        if self.reward_weight > 0 or self.value_weight > 0:
            logs['reward_loss'] = torch.mean(mse[:, :, -2].detach())
            logs['value_loss'] = torch.mean(mse[:, :, -1].detach())
        else:
            logs['reward_loss'] = torch.zeros(1, device=joined_inputs.device)
            logs['value_loss'] = torch.zeros(1, device=joined_inputs.device)

        logs['terminal_loss'] = cross_entropy.detach()

        self.ema_reconstruction = self.ema_reconstruction * self.ema_rate + torch.mean(mse).detach() * (
                1 - self.ema_rate)

        if self.data_parallel:
            # return dummy value
            logs = namedtuple('logs', logs.keys())(**logs)
            return reconstructed, feature, reconstruction_loss, loss_vq, loss_commit, logs
        else:
            return reconstructed, feature, reconstruction_loss, loss_vq, loss_commit, logs
    
    def ema_update(self, feature):
        self.model.codebook.ema_update(feature)

def update_exp_decaying_std_torch(minibatch_returns, prev_expectation, prev_std, decay_rate=0.999):
    # Convert input to tensors if they are not
    if not isinstance(minibatch_returns, torch.Tensor):
        minibatch_returns = torch.tensor(minibatch_returns)
    if not isinstance(prev_expectation, torch.Tensor):
        prev_expectation = torch.tensor(prev_expectation)
    if not isinstance(prev_std, torch.Tensor):
        prev_std = torch.tensor(prev_std)
    # Compute the updated expectation
    new_expectation = decay_rate * prev_expectation + (1 - decay_rate) * torch.mean(minibatch_returns)
    if torch.isnan(new_expectation):
        new_expectation = prev_expectation
    # Compute the updated variance
    minibatch_var = torch.var(minibatch_returns, unbiased=False)
    new_variance = decay_rate * prev_std ** 2 + (1 - decay_rate) * minibatch_var
    # Compute the updated standard deviation
    new_std = torch.sqrt(new_variance)
    if torch.isnan(new_std):
        new_std = prev_std
    return new_expectation, new_std






