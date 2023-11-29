import torch
import torch.nn as nn
from trajectory.models.transformers import *
from trajectory.models.ein import EinLinear

class TransformerPrior(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        if "data_parallel" in config:
            self.data_parallel = config.data_parallel
        else:
            self.data_parallel = False

        self.transition_dim = config.transition_dim
        self.observation_dim = config.observation_dim
        self.action_dim = config.action_dim
        # inputs to embed in additon to states
        self.n_embd = config.n_embd
        self.max_latents_length = config.max_sequence_length // config.latent_step * config.code_per_step

        # transformer for the policy
        self.tok_emb = nn.Embedding(config.K, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_latents_length, config.n_embd))
        self.state_emb = nn.Linear(config.observation_dim, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList([Block(config.n_embd, config.n_head, config.resid_pdrop,
                                           config.attn_pdrop, sequence_length=self.max_latents_length,
                                           causal=True, rotary=(config.position_embedding == "rotary"),
                                           kv_cache=True) for _ in
                                  range(config.n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.latent_step = config.latent_step
        self.code_per_step = config.code_per_step

        # output tensor is a concatenation of logits and q_value
        self.policy_head = nn.Linear(config.n_embd, config.K, bias=False)

        self.vocab_size = config.K
        self.embedding_dim = config.n_embd
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

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
        whitelist_weight_modules = (torch.nn.Linear, EinLinear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

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
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, state, targets=None, progress=0, mask=None, kv_cache=None):
        """
            idx : [ B x T ]
            state: [ B ]
        """

        state = state.to(device=self.pos_emb.device, dtype=torch.float32)
        ## [ B x T x embedding_dim ]

        if kv_cache is None:
            if not idx is None:
                b, t = idx.size()
                assert t <= self.max_latents_length, "Cannot forward, model block size is exhausted."
                token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
                token_embeddings = torch.cat(
                    [torch.zeros(size=(b, 1, self.embedding_dim)).to(token_embeddings), token_embeddings],
                    dim=1)
            else:
                b = state.size(0)
                t = 0
                token_embeddings = torch.zeros(size=(b, 1, self.embedding_dim)).to(state)
        else:
            b = idx.size(0)
            t = kv_cache[0].shape[1]
            token_embeddings = self.tok_emb(idx).to(state)

        ## [ 1 x T+1 x embedding_dim ]
        if kv_cache is None:
            position_embeddings = self.pos_emb[:, :t + 1, :]  # each position maps to a (learnable) vector
        else:
            position_embeddings = self.pos_emb[:, kv_cache[0].shape[1], :]
        ## [ B x 1 x embedding_dim]
        state_embeddings = self.state_emb(state)[:, None]
        ## [ B x T+1 x embedding_dim ]
        x = self.drop(token_embeddings + position_embeddings + state_embeddings)

        new_cache = []
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        for block, block_cache in zip(self.blocks, kv_cache):
            x, new_block_cache = block(x, kv_cache=block_cache)
            if not self.training:
                new_cache.append(new_block_cache)

        ## [ B x T+1 x embedding_dim ]
        x = self.ln_f(x)

        # if we are given some desired targets also calculate the loss
        if targets is not None:
            policy_logits = self.policy_head(x).reshape(b, t + 1, -1)
            # update return stats
            target_latent_codes = targets["codes"]
            policy_loss = F.cross_entropy(policy_logits.reshape(-1, self.vocab_size), target_latent_codes.reshape([-1]),
                                          reduction='none')

            logs = {"latent_policy_loss": policy_loss.detach(),
                    }

            if "weights" in targets:
                weights_selected = targets["weights"].gather(-1, target_latent_codes[:, :, None])
                policy_loss = torch.reshape(policy_loss, [b, t+1, -1]) * weights_selected

            if mask is not None:
                policy_loss = policy_loss.reshape(b, -1) * mask[:, :, 0]

            return policy_logits, policy_loss, new_cache, logs
        else:
            if kv_cache[0] is not None:
                policy_logits = self.policy_head(x).reshape(b, 1, -1)
            else:
                policy_logits = self.policy_head(x).reshape(b, t+1, -1)
            loss = torch.tensor(0.0, device=state.device)
            return policy_logits, loss, new_cache, {}
