import math

import torch
import torch.nn.functional as F
from torch import nn

MEMORY_FUSIONS = ("adaln_add", "prefix_tokens")


def sinusoidal_embedding(
    values: torch.Tensor, dim: int, max_period: float = 1000.0
) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = values.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class _Mlp(nn.Sequential):
    def __init__(self, hidden_size: int, mlp_ratio: float, dropout: float):
        inner = int(hidden_size * mlp_ratio)
        super().__init__(
            nn.Linear(hidden_size, inner),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_size),
        )


class _Attention(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("mem_hidden_size must be divisible by mem_num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.kv = nn.Linear(hidden_size, 2 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, q_len, hidden_size = x.shape
        kv_len = context.shape[1]
        q = self.q(x).view(batch_size, q_len, self.num_heads, self.head_dim)
        k, v = (
            self.kv(context)
            .view(batch_size, kv_len, 2, self.num_heads, self.head_dim)
            .unbind(dim=2)
        )
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = (~key_padding_mask)[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch_size, q_len, hidden_size)
        return self.out_proj(out)


class MemorySelfAttentionBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = _Attention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = _Mlp(hidden_size, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, key_padding_mask)
        return x + self.mlp(self.norm2(x))


class MemoryResamplerBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_kv = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = _Attention(hidden_size, num_heads, dropout)
        self.self_attn = MemorySelfAttentionBlock(hidden_size, num_heads, mlp_ratio, dropout)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        queries = queries + self.cross_attn(
            self.norm_q(queries), self.norm_kv(memory), key_padding_mask
        )
        return self.self_attn(queries, None)


class VisionTokenMemory(nn.Module):
    """History vision-token cache -> Transformer -> fixed-length tokens -> DiT condition.

    Input is the per-frame ViT (merger) tokens of the cached keyframes,
    ``feats (B, K, N, feature_dim)`` with ``age (B, K)`` in seconds before the
    current frame and ``mask (B, K)`` marking real (non-pad) frames.
    """

    def __init__(
        self,
        feature_dim: int,
        cond_dim: int,
        tokens_per_frame: int,
        hidden_size: int = 1024,
        depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_queries: int = 16,
        compressor_depth: int = 2,
        fusion: str = "adaln_add",
        dropout: float = 0.0,
    ):
        super().__init__()
        if fusion not in MEMORY_FUSIONS:
            raise ValueError(f"mem_fusion must be one of {MEMORY_FUSIONS}, got {fusion!r}")
        self.fusion = fusion
        self.hidden_size = hidden_size
        self.tokens_per_frame = tokens_per_frame
        self.in_norm = nn.LayerNorm(feature_dim, eps=1e-6)
        self.in_proj = nn.Linear(feature_dim, hidden_size)
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, tokens_per_frame, hidden_size))
        self.age_mlp = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.encoder = nn.ModuleList(
            [
                MemorySelfAttentionBlock(hidden_size, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.queries = nn.Parameter(torch.zeros(1, num_queries, hidden_size))
        self.compressor = nn.ModuleList(
            [
                MemoryResamplerBlock(hidden_size, num_heads, mlp_ratio, dropout)
                for _ in range(compressor_depth)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        if fusion == "adaln_add":
            self.out_proj = nn.Sequential(
                nn.Linear(num_queries * hidden_size, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
        else:
            self.out_proj = nn.Linear(hidden_size, cond_dim)
            self.token_type = nn.Parameter(torch.zeros(1, 1, cond_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.spatial_pos, std=0.02)
        nn.init.normal_(self.queries, std=0.02)
        last = self.out_proj[-1] if isinstance(self.out_proj, nn.Sequential) else self.out_proj
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        if self.fusion == "prefix_tokens":
            nn.init.normal_(self.token_type, std=0.02)

    @property
    def num_queries(self) -> int:
        return self.queries.shape[1]

    def compress(
        self,
        feats: torch.Tensor,
        age: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_frames, num_tokens, _ = feats.shape
        if num_tokens != self.tokens_per_frame:
            raise ValueError(
                f"memory features carry {num_tokens} tokens/frame; the module was "
                f"built for {self.tokens_per_frame} (mem_frame_pool**2)"
            )
        dtype = self.in_proj.weight.dtype
        x = self.in_proj(self.in_norm(feats.to(dtype)))
        age_emb = self.age_mlp(sinusoidal_embedding(age, 256).to(dtype))
        x = x + self.spatial_pos + age_emb.unsqueeze(2)
        x = x.reshape(batch_size, num_frames * num_tokens, self.hidden_size)
        valid = mask.to(dtype=torch.bool)
        valid = valid.unsqueeze(-1).expand(-1, -1, num_tokens).reshape(batch_size, -1)
        key_padding_mask = ~valid
        for block in self.encoder:
            x = block(x, key_padding_mask)
        queries = self.queries.expand(batch_size, -1, -1).to(dtype)
        for block in self.compressor:
            queries = block(queries, x, key_padding_mask)
        return self.out_norm(queries)

    def forward(
        self,
        feats: torch.Tensor,
        age: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns ``(B, cond_dim)`` for ``adaln_add`` or ``(B, M, cond_dim)`` for
        ``prefix_tokens``."""
        z = self.compress(feats, age, mask)
        if self.fusion == "adaln_add":
            return self.out_proj(z.flatten(1))
        return self.out_proj(z) + self.token_type
