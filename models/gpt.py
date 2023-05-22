import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cuda as cuda


class NewGELU(nn.Module):
    def forward(self, x):
        # https://github.com/karpathy/nanoGPT/blob/master/model.py#L19
        _sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        _x_pow_3 = torch.pow(x, 3.0)
        _tanh_inp = _sqrt_2_over_pi * (x + 0.044715 * _x_pow_3)
        return 0.5 * x * (1.0 + torch.tanh(_tanh_inp))


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class Attention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, D = x.shape
        H = self.heads
        dk = self.d_k

        q = self.q(x).reshape(B, N, H, dk).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, H, dk).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, H, dk).permute(0, 2, 1, 3)

        with cuda.sdp_kernel(enable_math=False):
            a = F.scaled_dot_product_attention(
                q.half(), k.half(), v.half(), is_causal=True
            ).float()

        a = a.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.fc(a)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.
    You can use this function to replace "F.gumbel_softmax".
    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)
    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    """

    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class Discretizer(nn.Module):
    def __init__(self, d_model, num_tokens):
        super().__init__()
        self.vocb = nn.Embedding(num_tokens, d_model, max_norm=1.0)
        self.prep = nn.Linear(d_model, num_tokens)

    def forward(self, x):
        return gumbel_softmax(self.prep(x), tau=1, dim=-1) @ self.vocb.weight


class Dictionary(nn.Sequential):
    def __init__(self, d_model, num_tokens):
        super().__init__(
            nn.Linear(d_model, d_model),
            NewGELU(),
            LayerNorm(d_model, bias=False),
            nn.Linear(d_model, num_tokens),
            nn.Softmax(dim=-1),
            nn.Linear(num_tokens, d_model),
        )


class Block(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.norm1 = LayerNorm(d_model, bias=False)
        self.norm2 = LayerNorm(d_model, bias=False)
        self.dict = Dictionary(d_model, d_ff)
        self.attn = Attention(d_model, heads)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.dict(self.norm2(x))
        return x


class GPTa(nn.Module):
    def __init__(self, d_model, heads, d_ff, num_layers, num_tokens, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, d_model, max_norm=1.0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model, max_norm=1.0)

        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model,
                    heads,
                    d_ff,
                )
                for _ in range(num_layers)
            ]
        )
        self.last = nn.Sequential(
            LayerNorm(d_model, bias=False),
            nn.Linear(d_model, num_tokens),
        )

    def param_sum(self, include_embeddings=False):
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = list(self.token_emb.parameters()) + list(self.pos_emb.parameters())
        emb_params = sum(p.numel() for p in emb_params if p.requires_grad)
        return params - emb_params if not include_embeddings else params

    def forward(self, x):
        B, N = x.shape
        assert N <= self.max_seq_len, "Sequence length exceeds model capacity"

        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(N, device=x.device))
        x = token_emb + pos_emb

        for i, block in enumerate(self.blocks):
            x = block(x)

        l = self.last(x)
        return l


def test():
    seq = 2**10
    model = GPTa(
        d_model=256,
        heads=8,
        d_ff=1024,
        num_layers=16,
        num_tokens=10000,
        max_seq_len=seq,
    ).cuda()
    print(model.param_sum())
    print(model.param_sum(include_embeddings=True))
    x = torch.randint(0, 10000, (1, seq), dtype=torch.long).cuda()
    logits = model(x)
    logits.sum().backward()
    print(logits.shape)


if __name__ == "__main__":
    test()
    input()