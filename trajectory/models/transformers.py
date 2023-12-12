# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from xformers.components.attention import ScaledDotProduct
import xformers.ops as xops

from xformers.factory import xFormerEncoderBlock, xFormerEncoderConfig



class Block(nn.Module):
    def __init__(self, n_embd, n_head, resid_pdrop, attn_pdrop, sequence_length, causal=True, rotary=False,
                 kv_cache=False):
        super().__init__()
        """
        if "ma_update" in config:
            sequence_length = int(config.block_size / config.transition_dim)
        else:
            sequence_length = int(config.max_sequence_length / config.latent_step)*config.code_per_step
        """

        self.sequence_length = sequence_length

        # if triton is installed, use FusedMLP, otherwise use MLP

        block_config = {
            "dim_model": n_embd,
            "residual_norm_style": "post",  # Optional, pre/post

            "multi_head_config": {
                "num_heads": n_head,
                "residual_dropout": resid_pdrop,
                "attention": {
                    "name": "scaled_dot_product",
                    "dropout": attn_pdrop,
                    "seq_len": sequence_length,
                    "num_rules": n_head,
                },
            },
            "feedforward_config": {
                "name": "MLP",
                "dropout": resid_pdrop,
                "activation": "gelu",
                "hidden_layer_multiplier": 4,
            },
        }
        if rotary:
            block_config["multi_head_config"]["attention"]["rotary"] = True
            block_config["multi_head_config"]["attention"]["rotary_dim"] = n_embd
        config = xFormerEncoderConfig(**block_config)
        self.block = xFormerEncoderBlock(config)
        self.causal = causal
        self.kv_cache = kv_cache
        self.attention_mask = torch.ones((sequence_length, sequence_length), device="cuda")
        if causal:
            self.attention_mask = torch.tril(self.attention_mask)

    def forward(self, x, kv_cache=None):
        # pad input to sequence length
        if self.block.patch_emb is not None:
            x = self.block.patch_emb(x)

        if self.block.pose_encoding is not None:
            x = self.block.pose_encoding(x)

            if hasattr(self.block, "embedding_projector"):
                x = self.block.embedding_projector(x)

        # Handle the optional input masking, differs on Q, K, V
        if kv_cache is None:
            q, k, v = x, x, x
        else:
            assert x.size(1) == 1, "Only used for autoregressive decoding"
            if kv_cache.size(1) == 1:
                kv_cache = kv_cache.repeat(x.size(0), 1, 1)
            q, k, v = x, torch.cat([kv_cache, x], dim=1), torch.cat([kv_cache, x], dim=1)

        new_cache = k
        # Pre/Post norms and residual paths are already handled
        if kv_cache is None:
            x += self.block.wrap_att.sublayer.layer(q, k, v, att_mask=self.attention_mask)
        else:
            x += self.block.wrap_att.sublayer.layer(q, k, v, att_mask=None)

        x = self.block.wrap_att.norm(x)
        x = self.block.wrap_ff(inputs=[x])

        # Optional simplicial embeddings
        if self.block.simplicial_embedding is not None:
            x = self.block.simplicial_embedding(x)
        if self.kv_cache:
            return x, new_cache
        else:
            return x