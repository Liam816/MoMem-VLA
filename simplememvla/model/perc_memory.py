"""Perception memory v1: a fixed-length vision-token queue with learned eviction.

Every front-camera keyframe (native ViT-merger tokens, e.g. 8x8 x 2560) enters a
FIFO window of the ``W`` most recent frames. The frame pushed out of the window
joins a queue of ``K`` frames; once the queue is full, the ``K + 1`` candidates
(queue plus newcomer) are ranked by ``EvictScorer`` and the lowest one is
dropped -- possibly the newcomer itself. ``Q ∪ W`` (at most ``K + W`` frames)
is what ``MemReader`` reads: a Perceiver resampler that compresses it into one
vector ``c_mem`` added to the DiT's adaLN condition (zero-initialised, so the
model starts identical to the memory-free one), or -- the ``prefix_tokens``
ablation -- one DiT token per read token.

The queue stores native tokens; scoring and reading pool them separately, so
read resolution can change without touching the write path. Eviction is a hard,
deterministic choice, so the action loss never reaches the scorer: it is
trained by a pairwise ranking loss against offline teacher utilities.
"""

import math
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from simplememvla.model.memory import MemoryResamplerBlock, sinusoidal_embedding

EVICT_MODES = ("learned", "fifo", "uniform", "random", "oracle")
READ_FUSIONS = ("adaln_add", "prefix_tokens")
AGE_FREQUENCY_DIM = 256


def pool_tokens(x: torch.Tensor, pool: int) -> torch.Tensor:
    """``(..., G*G, D)`` -> ``(..., pool*pool, D)`` by 2-D average pooling (square grid)."""
    *lead, num_tokens, dim = x.shape
    grid = math.isqrt(num_tokens)
    if grid * grid != num_tokens:
        raise ValueError(f"{num_tokens} tokens/frame are not a square grid")
    if grid == pool:
        return x
    y = x.reshape(-1, grid, grid, dim).permute(0, 3, 1, 2).float()
    y = F.adaptive_avg_pool2d(y, (pool, pool))
    return y.flatten(2).transpose(1, 2).reshape(*lead, pool * pool, dim).to(x.dtype)


@dataclass
class PercMemState:
    """Batched memory state with ``K + W`` slots per row.

    Slots ``[:K]`` are the queue, kept sorted by time with empty slots last;
    slots ``[K:]`` are the window, oldest first and right-aligned. Every queue
    frame is older than every window frame, so slot order is time order.
    ``time`` is in seconds (any origin, increasing); ``src`` is the caller's
    frame id (the history index offline) and ``-1`` for empty slots.
    ``summary`` caches the scorer's per-frame summary, computed once at write.
    """

    tokens: torch.Tensor  # (B, K+W, P_store, D) native merger tokens
    summary: torch.Tensor  # (B, K+W, H) scorer frame summaries
    time: torch.Tensor  # (B, K+W) float32
    src: torch.Tensor  # (B, K+W) long
    valid: torch.Tensor  # (B, K+W) bool

    def where(self, cond: torch.Tensor, other: "PercMemState") -> "PercMemState":
        """Per-row select: ``self`` where ``cond (B,)`` else ``other``."""
        out = {}
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            out[f.name] = torch.where(cond.view(-1, *([1] * (a.ndim - 1))), a, b)
        return PercMemState(**out)


class EvictScorer(nn.Module):
    """Per-candidate keep logit; lower scores are evicted first.

    Each frame is pooled to ``pool x pool`` tokens, projected and
    attention-pooled to one summary; frame age, the newcomer flag and a
    stop-grad redundancy cue (max cosine similarity to another candidate) are
    added, and a Transformer lets candidates compare against each other.
    No text input: writing is task-agnostic. ``summarize`` depends on one
    frame only, so callers run it once per frame and cache the result.
    """

    def __init__(
        self,
        feature_dim: int,
        pool: int = 4,
        hidden_size: int = 256,
        num_heads: int = 4,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_redund: bool = True,
    ):
        super().__init__()
        self.pool = int(pool)
        self.hidden_size = int(hidden_size)
        self.in_norm = nn.LayerNorm(feature_dim, eps=1e-6)
        self.in_proj = nn.Linear(feature_dim, hidden_size)
        self.pool_query = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.pool_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.age_proj = nn.Linear(AGE_FREQUENCY_DIM, hidden_size)
        self.new_embed = nn.Embedding(2, hidden_size)
        self.redund_proj = nn.Linear(1, hidden_size) if use_redund else None
        layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_heads,
            int(hidden_size * mlp_ratio),
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.out = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.pool_query, std=0.02)

    def summarize(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, C, P_store, D)`` -> ``(B, C, H)``."""
        batch_size, num_cands = tokens.shape[:2]
        x = pool_tokens(tokens, self.pool).to(self.in_proj.weight.dtype)
        x = self.in_proj(self.in_norm(x)).reshape(batch_size * num_cands, self.pool**2, -1)
        query = self.pool_query.expand(x.shape[0], -1, -1).to(x.dtype)
        f, _ = self.pool_attn(query, x, x, need_weights=False)
        return f.reshape(batch_size, num_cands, -1)

    @staticmethod
    def redundancy(summary: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Max cosine similarity of each candidate to any other valid one."""
        f = F.normalize(summary.detach().float(), dim=-1)
        sim = f @ f.transpose(1, 2)
        eye = torch.eye(sim.shape[-1], dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(eye | ~valid[:, None, :], -1.0)
        return sim.amax(dim=-1)

    def forward(
        self,
        f: torch.Tensor,
        age: torch.Tensor,
        is_new: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Raw fp32 logits ``(B, C)`` from summaries ``f (B, C, H)``; invalid entries are meaningless."""
        f = f.to(self.age_proj.weight.dtype)
        x = f + self.age_proj(sinusoidal_embedding(age, AGE_FREQUENCY_DIM).to(f.dtype))
        x = x + self.new_embed(is_new.long())
        if self.redund_proj is not None:
            x = x + self.redund_proj(self.redundancy(f, valid)[..., None].to(f.dtype))
        if torch.is_grad_enabled() and self.training:
            # Replay runs this once per eviction event; keep only the inputs.
            x = checkpoint(self.encoder, x, None, ~valid, use_reentrant=False)
        else:
            x = self.encoder(x, src_key_padding_mask=~valid)
        return self.out(self.out_norm(x)).squeeze(-1).float()


class MemReader(nn.Module):
    """Reads the kept frames: per-token embedding, then Perceiver resampler -> ``c_mem``.

    ``m = LN(Linear(LN(x))) + pos + MLP(sinusoid(age))`` for every pooled token
    of every valid frame. ``adaln_add``: ``M`` learned queries cross-attend to
    ``m`` (key padding masks empty slots) for ``depth`` layers, then
    ``flatten -> Linear -> SiLU -> Linear`` (last layer zero-initialised) gives
    ``c_mem (B, cond_dim)``. ``prefix_tokens``: ``m`` itself is projected to
    ``cond_dim`` and returned as DiT prefix tokens with their valid mask.
    """

    def __init__(
        self,
        feature_dim: int,
        cond_dim: int,
        tokens_per_frame: int = 16,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_queries: int = 8,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        fusion: str = "adaln_add",
        dropout: float = 0.0,
    ):
        super().__init__()
        if fusion not in READ_FUSIONS:
            raise ValueError(f"mem_fusion must be one of {READ_FUSIONS}, got {fusion!r}")
        self.fusion = fusion
        self.tokens_per_frame = int(tokens_per_frame)
        self.hidden_size = int(hidden_size)
        self.in_norm = nn.LayerNorm(feature_dim, eps=1e-6)
        self.in_proj = nn.Linear(feature_dim, hidden_size)
        self.proj_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.spatial_pos = nn.Parameter(torch.empty(1, 1, tokens_per_frame, hidden_size))
        self.age_mlp = nn.Sequential(
            nn.Linear(AGE_FREQUENCY_DIM, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        if fusion == "adaln_add":
            self.queries = nn.Parameter(torch.empty(1, num_queries, hidden_size))
            self.resampler = nn.ModuleList(
                [
                    MemoryResamplerBlock(hidden_size, num_heads, mlp_ratio, dropout)
                    for _ in range(depth)
                ]
            )
            self.out_norm = nn.LayerNorm(hidden_size, eps=1e-6)
            self.out_proj = nn.Sequential(
                nn.Linear(num_queries * hidden_size, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
            nn.init.normal_(self.queries, std=0.02)
            nn.init.zeros_(self.out_proj[-1].weight)
            nn.init.zeros_(self.out_proj[-1].bias)
        else:
            self.out_proj = nn.Linear(hidden_size, cond_dim)
            self.token_type = nn.Parameter(torch.empty(1, 1, cond_dim))
            nn.init.normal_(self.token_type, std=0.02)
        nn.init.normal_(self.spatial_pos, std=0.02)

    def embed(self, feats: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        """``(B, F, P, D)``, ``(B, F)`` -> ``m (B, F, P, hidden)``."""
        dtype = self.in_proj.weight.dtype
        m = self.proj_norm(self.in_proj(self.in_norm(feats.to(dtype))))
        age_emb = self.age_mlp(sinusoidal_embedding(age, AGE_FREQUENCY_DIM).to(dtype))
        return m + self.spatial_pos + age_emb[:, :, None, :]

    def forward(
        self, feats: torch.Tensor, age: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_frames, num_tokens, _ = feats.shape
        if num_tokens != self.tokens_per_frame:
            raise ValueError(
                f"reader built for {self.tokens_per_frame} tokens/frame, got {num_tokens}"
            )
        valid = valid.to(dtype=torch.bool)
        safe = feats.masked_fill(~valid[:, :, None, None], 0)
        m = self.embed(safe, age.masked_fill(~valid, 0))
        m = m.reshape(batch_size, num_frames * num_tokens, self.hidden_size)
        token_valid = valid[:, :, None].expand(-1, -1, num_tokens).reshape(batch_size, -1)
        if self.fusion == "prefix_tokens":
            tokens = self.out_proj(m) + self.token_type
            return tokens.masked_fill(~token_valid[:, :, None], 0), token_valid
        z = self.queries.expand(batch_size, -1, -1).to(m.dtype)
        for block in self.resampler:
            z = block(z, m, ~token_valid)
        return self.out_proj(self.out_norm(z).flatten(1))


class PerceptionMemory(nn.Module):
    """Fixed-length perception memory: ``write`` maintains the queue, ``read`` -> DiT.

    Offline (training / open loop) ``forward`` replays the whole history through
    ``write`` in time order, so training sees exactly the queue a rollout would.
    Online, keep a ``PercMemState`` from ``init_state`` and call ``write`` once
    per keyframe and ``read`` once per decision.
    """

    def __init__(
        self,
        feature_dim: int,
        cond_dim: int,
        window_frames: int = 4,
        queue_frames: int = 28,
        evict_mode: str = "learned",
        score_pool: int = 4,
        read_pool: int = 4,
        scorer_hidden_size: int = 256,
        scorer_num_heads: int = 4,
        scorer_depth: int = 2,
        scorer_mlp_ratio: float = 4.0,
        scorer_dropout: float = 0.1,
        scorer_use_redund: bool = True,
        reader_hidden_size: int = 512,
        reader_num_heads: int = 8,
        reader_num_queries: int = 8,
        reader_depth: int = 2,
        reader_mlp_ratio: float = 4.0,
        fusion: str = "adaln_add",
        rank_margin: float = 0.1,
        oracle_prob: float = 0.0,
    ):
        super().__init__()
        if window_frames < 1 or queue_frames < 1:
            raise ValueError("perc_mem_window and perc_mem_queue must be >= 1")
        if evict_mode not in EVICT_MODES:
            raise ValueError(f"perc_mem_evict must be one of {EVICT_MODES}, got {evict_mode!r}")
        self.window_frames = int(window_frames)
        self.queue_frames = int(queue_frames)
        self.evict_mode = evict_mode
        self.read_pool = int(read_pool)
        self.fusion = fusion
        self.rank_margin = float(rank_margin)
        self.oracle_prob = float(oracle_prob)
        self.scorer = EvictScorer(
            feature_dim,
            pool=score_pool,
            hidden_size=scorer_hidden_size,
            num_heads=scorer_num_heads,
            depth=scorer_depth,
            mlp_ratio=scorer_mlp_ratio,
            dropout=scorer_dropout,
            use_redund=scorer_use_redund,
        )
        self.reader = MemReader(
            feature_dim,
            cond_dim,
            tokens_per_frame=self.read_pool**2,
            hidden_size=reader_hidden_size,
            num_heads=reader_num_heads,
            num_queries=reader_num_queries,
            depth=reader_depth,
            mlp_ratio=reader_mlp_ratio,
            fusion=fusion,
        )

    @property
    def num_slots(self) -> int:
        return self.queue_frames + self.window_frames

    def init_state(
        self,
        batch_size: int,
        tokens_per_frame: int,
        feature_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> PercMemState:
        s = self.num_slots
        h = self.scorer.hidden_size
        return PercMemState(
            tokens=torch.zeros(batch_size, s, tokens_per_frame, feature_dim, device=device, dtype=dtype),
            summary=torch.zeros(batch_size, s, h, device=device, dtype=self.scorer.in_proj.weight.dtype),
            time=torch.zeros(batch_size, s, device=device, dtype=torch.float32),
            src=torch.full((batch_size, s), -1, device=device, dtype=torch.long),
            valid=torch.zeros(batch_size, s, device=device, dtype=torch.bool),
        )

    def reset(self, state: PercMemState, rows: torch.Tensor) -> PercMemState:
        """Empty the rows selected by the bool mask ``rows (B,)`` (e.g. finished envs)."""
        b, _, p, d = state.tokens.shape
        empty = self.init_state(b, p, d, state.tokens.device, state.tokens.dtype)
        return empty.where(rows.to(state.valid.device), state)

    def _choose_victim(
        self,
        scores: torch.Tensor | None,
        cand_time: torch.Tensor,
        cand_valid: torch.Tensor,
        cand_util: torch.Tensor | None,
    ) -> torch.Tensor:
        """Index in ``[0, K]`` of the candidate to evict (``K`` = the newcomer)."""
        inf = torch.finfo(torch.float32).max
        mode = self.evict_mode
        if mode == "fifo":
            return cand_time.masked_fill(~cand_valid, inf).argmin(dim=1)
        if mode == "random":
            noise = torch.rand(cand_valid.shape, device=cand_valid.device)
            return noise.masked_fill(~cand_valid, 2.0).argmin(dim=1)
        if mode == "uniform":
            # Evict the interior frame whose two neighbours are closest together;
            # the oldest and the newest candidate are never evicted.
            order = cand_time.masked_fill(~cand_valid, inf).argsort(dim=1, stable=True)
            t = cand_time.gather(1, order)
            n = cand_valid.sum(dim=1, keepdim=True)
            pos = torch.arange(t.shape[1], device=t.device)[None]
            gap = torch.full_like(t, inf)
            gap[:, 1:-1] = t[:, 2:] - t[:, :-2]
            pick = gap.masked_fill((pos < 1) | (pos >= n - 1), inf).argmin(dim=1)
            return order.gather(1, pick[:, None]).squeeze(1)
        if mode == "oracle":
            if cand_util is None:
                raise ValueError("perc_mem_evict='oracle' needs mem_utility labels")
            return cand_util.float().masked_fill(~cand_valid, inf).argmin(dim=1)
        learned = scores.detach().masked_fill(~cand_valid, inf).argmin(dim=1)
        if self.training and self.oracle_prob > 0.0 and cand_util is not None:
            # Teacher-forced evictions keep early-training queues realistic.
            oracle = cand_util.float().masked_fill(~cand_valid, inf).argmin(dim=1)
            use_oracle = torch.rand(oracle.shape, device=oracle.device) < self.oracle_prob
            return torch.where(use_oracle, oracle, learned)
        return learned

    def _rank_loss(
        self,
        scores: torch.Tensor,
        cand_util: torch.Tensor,
        cand_valid: torch.Tensor,
        rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sum and count of ``log(1 + exp(-(s_i - s_j)))`` over pairs with ``u_i - u_j > margin``."""
        u = cand_util.float()
        pair = (u[:, :, None] - u[:, None, :]) > self.rank_margin
        pair &= cand_valid[:, :, None] & cand_valid[:, None, :] & rows[:, None, None]
        diff = scores[:, :, None] - scores[:, None, :]
        loss = F.softplus(-diff).masked_fill(~pair, 0.0)
        return loss.sum(), pair.sum()

    def write(
        self,
        state: PercMemState,
        frame: torch.Tensor,
        time: torch.Tensor,
        src: torch.Tensor | None = None,
        active: torch.Tensor | None = None,
        utility: torch.Tensor | None = None,
    ) -> tuple[PercMemState, torch.Tensor, torch.Tensor]:
        """Write one keyframe per row and return ``(state, rank_loss_sum, num_pairs)``.

        ``frame (B, P_store, D)`` native tokens, ``time (B,)`` seconds, ``src
        (B,)`` frame id, ``active (B,)`` rows that observe a keyframe at this
        step. ``utility (B, N)`` maps a ``src`` id to its teacher value at this
        step; it is only needed for ``oracle`` and the rank loss.
        """
        batch_size = frame.shape[0]
        k = self.queue_frames
        device = frame.device
        zero = torch.zeros((), device=device)
        if active is None:
            active = torch.ones(batch_size, dtype=torch.bool, device=device)
        if src is None:
            src = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        time = time.to(device=device, dtype=torch.float32)
        old = state
        train_scorer = (
            self.training
            and utility is not None
            and torch.is_grad_enabled()
            and any(p.requires_grad for p in self.scorer.parameters())
        )
        with torch.set_grad_enabled(train_scorer):
            summary = self.scorer.summarize(frame[:, None])[:, 0]

        # 1) Window: shift left, newcomer at the end; slot K is pushed out.
        def push(x: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
            return torch.cat([x[:, :k], x[:, k + 1 :], new[:, None].to(x.dtype)], dim=1)

        pushed = PercMemState(
            tokens=push(old.tokens, frame),
            summary=push(old.summary, summary),
            time=push(old.time, time),
            src=push(old.src, src),
            valid=push(old.valid, torch.ones_like(active)),
        )
        cand_ok = active & old.valid[:, k]

        # 2) Queue: append while not full, otherwise evict one of K + 1.
        q_valid = old.valid[:, :k]
        full = q_valid.all(dim=1)
        slot = (~q_valid).to(torch.uint8).argmax(dim=1)  # first empty slot
        loss_sum, num_pairs = zero, zero.long()
        events = cand_ok & full
        if bool(events.any()):
            # Candidates: the K queue slots plus the frame leaving the window (slot K).
            cand_summary, cand_time, cand_src = (x[:, : k + 1] for x in (old.summary, old.time, old.src))
            cand_valid = old.valid[:, : k + 1]
            is_new = torch.zeros_like(cand_valid)
            is_new[:, k] = True
            cand_util = None
            if utility is not None:
                cand_util = utility.to(device).gather(1, cand_src.clamp_min(0))
            scores = None
            if self.evict_mode == "learned" or train_scorer:
                with torch.set_grad_enabled(train_scorer):
                    scores = self.scorer(cand_summary, time[:, None] - cand_time, is_new, cand_valid)
            if train_scorer:
                loss_sum, num_pairs = self._rank_loss(scores, cand_util, cand_valid, events)
            victim = self._choose_victim(scores, cand_time, cand_valid, cand_util)
            slot = torch.where(full, victim, slot)
        insert = cand_ok & (slot < k)  # victim == K: the newcomer itself is dropped
        slot = slot.clamp_max(k - 1)

        def place(x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            idx = slot.view(-1, *([1] * (x.ndim - 1))).expand(-1, 1, *x.shape[2:])
            updated = x.scatter(1, idx, value[:, None].to(x.dtype))
            return torch.where(insert.view(-1, *([1] * (x.ndim - 1))), updated, x)

        q = {
            name: place(getattr(pushed, name)[:, :k], getattr(old, name)[:, k])
            for name in ("tokens", "summary", "time", "src", "valid")
        }
        # Keep the queue in time order (empty slots last).
        order = q["time"].masked_fill(~q["valid"], float("inf")).argsort(dim=1, stable=True)
        for name, x in q.items():
            idx = order.view(*order.shape, *([1] * (x.ndim - 2))).expand_as(x)
            q[name] = x.gather(1, idx)
        new = PercMemState(
            **{name: torch.cat([q[name], getattr(pushed, name)[:, k:]], dim=1) for name in q}
        )
        return new.where(active, old), loss_sum, num_pairs

    def read(
        self, state: PercMemState, now: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``c_mem (B, cond_dim)``, or prefix tokens and their mask for ``prefix_tokens``."""
        age = now.to(state.time.device, torch.float32)[:, None] - state.time
        feats = pool_tokens(state.tokens, self.read_pool)
        return self.reader(feats, age.masked_fill(~state.valid, 0), state.valid)

    def replay(
        self,
        feats: torch.Tensor,
        age: torch.Tensor,
        mask: torch.Tensor,
        utility: torch.Tensor | None = None,
    ) -> tuple[PercMemState, torch.Tensor | None]:
        """Write a right-padded, oldest-first history keyframe by keyframe.

        ``feats (B, N, P_store, D)``, ``age (B, N)`` seconds before the current
        frame, ``mask (B, N)`` valid keyframes, ``utility (B, N, N)`` with row
        ``i`` the teacher value of every keyframe when keyframe ``i`` is written.
        Returns the final state (``src`` = history index) and the mean rank
        loss, which is ``None`` unless the scorer is being trained.
        """
        batch_size, num_frames, num_tokens, feature_dim = feats.shape
        device = feats.device
        mask = mask.to(device=device, dtype=torch.bool)
        time = -age.to(device).float()  # current frame at 0, older frames negative
        state = self.init_state(batch_size, num_tokens, feature_dim, device, feats.dtype)
        loss_sum = torch.zeros((), device=device)
        num_pairs = torch.zeros((), device=device, dtype=torch.long)
        for i in range(num_frames):
            state, step_sum, step_pairs = self.write(
                state,
                feats[:, i],
                time[:, i],
                src=torch.full((batch_size,), i, dtype=torch.long, device=device),
                active=mask[:, i],
                utility=None if utility is None else utility[:, i],
            )
            loss_sum = loss_sum + step_sum
            num_pairs = num_pairs + step_pairs
        rank_loss = None
        trainable = [p for p in self.scorer.parameters() if p.requires_grad]
        if self.training and utility is not None and torch.is_grad_enabled() and trainable:
            # The zero term keeps every scorer parameter in the graph (DDP) even
            # when no candidate pair passed the margin.
            rank_loss = loss_sum / num_pairs.clamp_min(1) + sum(p.sum() for p in trainable) * 0.0
        return state, rank_loss

    def forward(
        self,
        feats: torch.Tensor,
        age: torch.Tensor,
        mask: torch.Tensor,
        utility: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]:
        """``replay`` then ``read``: ``(c_mem or (tokens, mask), rank_loss)``.

        Recurrent-memory output, once added, would merge with this read here.
        """
        state, rank_loss = self.replay(feats, age, mask, utility)
        return self.read(state, torch.zeros(feats.shape[0], device=feats.device)), rank_loss
