"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from exercise_helpers import todo


def causal_mask(size, device=None):
    """Boolean [T,T] mask: True permits attending to that key (including self)."""
    # TODO causal_mask: Return a boolean lower-triangular [size,size] mask, including the diagonal.
    # Keep the supplied shape, dtype and device. Choose the tril diagonal
    # that permits the current token and all earlier tokens, never future tokens.
    return torch.ones(size, size, dtype=torch.bool, device=device).tril(
        diagonal=0)

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # The student's mask is used by BOTH SDPA and the readable manual path.
        self.register_buffer("bias", causal_mask(config.block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.bias[:T, :T], dropout_p=self.dropout if self.training else 0)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(~self.bias[:T, :T], float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

# nanoGPT model dimensions for training from scratch.
GPT2_PRESETS = {
    'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
    'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
    'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
    'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
}


def resolve_model_preset(config):
    name=config.get('model_preset','')
    if not name:
        return dict(config)  # tiny CPU exercises and the nanoGPT Shakespeare preset
    if name not in GPT2_PRESETS:
        raise ValueError(f'Unknown nanoGPT model preset: {name}')
    return {**config, **GPT2_PRESETS[name]}


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:block_size, :block_size]


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type,
                             optimizer_name='adamw', muon_learning_rate=1e-3,
                             muon_momentum=.95,
                             sgdm_learning_rate=.1, sgdm_momentum=.9, sgdm_nesterov=False):
        param_dict = {name: p for name, p in self.named_parameters() if p.requires_grad}
        def parameter_groups(parameters):
            return [
                {'params': [p for p in parameters if p.ndim >= 2], 'weight_decay': weight_decay},
                {'params': [p for p in parameters if p.ndim < 2], 'weight_decay': 0.0},
            ]
        all_groups = parameter_groups(list(param_dict.values()))
        # This fused-AdamW selection is retained from vanilla nanoGPT.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        if optimizer_name == 'adamw':
            # TODO optimizer_adamw: Set optimizers to a list containing AdamW(all_groups) with the supplied LR, betas and extra_args.
            # Use ONLY torch.optim.AdamW. Keep all_groups, betas and extra_args.
            # Fill lr from this function's learning_rate argument; do not hard-code,
            # rescale it, implement an optimizer, or import an optimizer from another file.
            optimizers = [torch.optim.AdamW(
                all_groups, lr=learning_rate, betas=betas, **extra_args)]
        elif optimizer_name == 'sgdm':
            # TODO optimizer_sgdm: Set optimizers to a list containing SGD(all_groups) with the supplied SGDM LR, momentum and Nesterov setting.
            # Use ONLY torch.optim.SGD (momentum SGD), not Adam or a custom class.
            # Fill lr from sgdm_learning_rate. Keep the supplied momentum/Nesterov
            # settings and parameter groups; do not create an auxiliary optimizer.
            optimizers = [torch.optim.SGD(
                all_groups, lr=sgdm_learning_rate,
                momentum=sgdm_momentum, nesterov=sgdm_nesterov)]
        elif optimizer_name == 'muon':
            if not hasattr(torch.optim, 'Muon'):
                raise RuntimeError('This course version requires PyTorch >= 2.9 with torch.optim.Muon')
            hidden = [p for name, p in param_dict.items() if name.startswith('transformer.h.') and p.ndim == 2]
            hidden_ids = {id(p) for p in hidden}
            other = [p for p in param_dict.values() if id(p) not in hidden_ids]
            other_groups = parameter_groups(other)
            # Muon handles hidden matrices; AdamW handles embeddings, tied head, norms and biases.
            # Keep the documented checkpoint/log order: auxiliary AdamW, then Muon.
            # TODO optimizer_muon: Construct auxiliary AdamW(other_groups) and torch.optim.Muon(hidden, adjust_lr_fn="original") using the supplied LRs/momentum/weight decay; set optimizers=[auxiliary, muon].
            # Use ONLY native torch.optim.Muon plus the supplied auxiliary AdamW.
            # Fill Muon's lr and momentum from muon_learning_rate and muon_momentum.
            # Keep hidden/other_groups, auxiliary LR/betas, weight_decay, extra_args,
            # adjust_lr_fn='original', and list order EXACTLY as provided.
            # Do not implement Newton-Schulz or an optimizer in another module, change
            # parameter membership, use a fallback optimizer, or add extra updates.
            auxiliary = torch.optim.AdamW(other_groups, lr=learning_rate, betas=betas, **extra_args)
            muon = torch.optim.Muon(
                hidden, lr=muon_learning_rate, momentum=muon_momentum,
                weight_decay=weight_decay, adjust_lr_fn='original')
            optimizers = [auxiliary, muon]
        else:
            raise ValueError('optimizer_name must be adamw, sgdm or muon')

        assigned = [id(p) for opt in optimizers for group in opt.param_groups for p in group['params']]
        if len(assigned) != len(set(assigned)) or set(assigned) != {id(p) for p in param_dict.values()}:
            raise ValueError('Every trainable parameter must belong to exactly one optimizer')
        for opt in optimizers:
            for group in opt.param_groups:
                group['base_lr'] = group['lr']
        return optimizers

    def estimate_mfu(self, fwdbwd_per_iter, dt, peak_tflops=None):
        """Estimate MFU using the GPU's dense peak TFLOPS at the chosen precision."""
        if peak_tflops is None or not all(math.isfinite(x) and x > 0
                                         for x in (fwdbwd_per_iter, dt, peak_tflops)):
            raise ValueError('Supply positive batch count, elapsed seconds and a verified dense peak_tflops')
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # Normalize only against a caller-supplied, verified peak for this hardware/precision.
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = peak_tflops * 1e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
