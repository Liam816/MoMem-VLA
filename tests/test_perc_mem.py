"""CPU unit tests for the perception memory (simplememvla/model/perc_memory.py).

    CUDA_VISIBLE_DEVICES='' python tests/test_perc_mem.py
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

torch.set_num_threads(4)

from simplememvla.data.collator import pad_memory
from simplememvla.model.dit_action_head import DiTActionHead
from simplememvla.model.perc_memory import PerceptionMemory, pool_tokens

D, P, W, K = 32, 16, 4, 6  # native 4x4 tokens/frame; score/read pool to 2x2
COND = 64


def make_memory(mode: str = "learned", fusion: str = "adaln_add", **kwargs) -> PerceptionMemory:
    torch.manual_seed(0)
    mem = PerceptionMemory(
        feature_dim=D,
        cond_dim=COND,
        window_frames=W,
        queue_frames=K,
        evict_mode=mode,
        score_pool=2,
        read_pool=2,
        scorer_hidden_size=16,
        scorer_num_heads=2,
        scorer_dropout=0.0,
        reader_hidden_size=16,
        reader_num_heads=2,
        reader_num_queries=2,
        fusion=fusion,
        **kwargs,
    )
    return mem.eval()


def history(n: int, batch: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(batch, n, P, D, generator=g)
    age = (n - 1 - torch.arange(n, dtype=torch.float32)).expand(batch, n) * 0.5
    mask = torch.ones(batch, n, dtype=torch.bool)
    return feats, age, mask


def kept(state, row: int = 0) -> list[int]:
    return sorted(state.src[row][state.valid[row]].tolist())


class PercMemTest(unittest.TestCase):
    def test_pool_tokens(self):
        x = torch.arange(16.0).view(1, 16, 1)
        self.assertEqual(pool_tokens(x, 2).flatten().tolist(), [2.5, 4.5, 10.5, 12.5])
        self.assertIs(pool_tokens(x, 4), x)

    def test_fixed_length(self):
        mem = make_memory("learned")
        state = mem.init_state(2, P, D)
        shape = state.tokens.shape
        for t in range(2000):
            frame = torch.randn(2, P, D)
            state, _, _ = mem.write(state, frame, torch.full((2,), t * 0.5), src=torch.full((2,), t))
        self.assertEqual(state.tokens.shape, shape)
        self.assertEqual(int(state.valid[:, :K].sum(1).min()), K)
        self.assertEqual(int(state.valid[:, K:].sum(1).min()), W)
        self.assertEqual(state.src[0, K:].tolist(), list(range(2000 - W, 2000)))

    def test_fifo_keeps_most_recent(self):
        state, _ = make_memory("fifo").replay(*history(50))
        self.assertEqual(kept(state), list(range(50 - K - W, 50)))

    def test_uniform_spacing(self):
        mem = make_memory("uniform")
        mem.queue_frames = 12
        state, _ = mem.replay(*history(200))
        t = torch.tensor(kept(state), dtype=torch.float32)
        # The newest queue frame was just admitted (the newcomer is never the
        # victim), so it sits one step from the window; spacing is about the rest.
        gaps = t[1:11] - t[:10]
        self.assertLessEqual(float(gaps.max() / gaps.min()), 2.0)
        self.assertEqual(int(t[0]), 0)  # the oldest frame is never evicted

    def test_short_history_keeps_everything(self):
        for mode in ("learned", "fifo", "uniform", "random"):
            state, _ = make_memory(mode).replay(*history(K + W))
            self.assertEqual(kept(state), list(range(K + W)), mode)

    def test_queue_stores_native_tokens(self):
        feats, age, mask = history(20)
        state, _ = make_memory("fifo").replay(feats, age, mask)
        self.assertEqual(tuple(state.tokens.shape[2:]), (P, D))
        for slot in range(K + W):
            self.assertTrue(torch.equal(state.tokens[0, slot], feats[0, state.src[0, slot]]))

    def test_padding_and_batch_consistency(self):
        """A padded batch replays exactly like each history on its own."""
        mem = make_memory("learned")
        lens = (40, 23, 7)
        hists = [history(n, seed=i) for i, n in enumerate(lens)]
        batch = pad_memory([h[0][0] for h in hists], [h[1][0] for h in hists])
        state_b, _ = mem.replay(batch["mem_feats"], batch["mem_age"], batch["mem_mask"])
        for row, h in enumerate(hists):
            state_1, _ = mem.replay(*h)
            self.assertEqual(kept(state_b, row), kept(state_1), f"row {row}")

    def test_online_write_matches_replay(self):
        mem = make_memory("learned")
        feats, age, mask = history(60)
        offline, _ = mem.replay(feats, age, mask)
        state = mem.init_state(1, P, D, dtype=feats.dtype)
        for i in range(60):
            state, _, _ = mem.write(state, feats[:, i], -age[:, i], src=torch.tensor([i]))
        self.assertEqual(kept(state), kept(offline))
        self.assertTrue(torch.equal(state.tokens, offline.tokens))

    def test_zero_init_c_mem(self):
        c_mem, _ = make_memory("learned")(*history(30))
        self.assertEqual(tuple(c_mem.shape), (1, COND))
        self.assertTrue(bool((c_mem == 0).all()))

    def test_zero_init_dit_equivalence(self):
        """With the zero-initialised reader the DiT output equals the memory-free one."""
        torch.manual_seed(0)
        dit = DiTActionHead(
            vlm_hidden_size=32, action_dim=4, action_horizon=5, rope_theta=10000.0,
            mrope_section=[3, 3, 2], partial_rotary_factor=0.25,
            hidden_size=COND, depth=2, num_heads=1, use_proprio=True, state_dim=3,
        ).eval()
        c_mem, _ = make_memory("learned")(*history(30, batch=2))
        b, length = 2, 7
        kwargs = dict(
            condition_hidden_states=torch.randn(b, length, 32),
            noisy_actions=torch.randn(b, 5, 4),
            timesteps=torch.rand(b),
            condition_position_ids=torch.arange(length).expand(3, b, length),
            condition_attention_mask=torch.ones(b, length, dtype=torch.long),
            state=torch.randn(b, 3),
        )
        with torch.no_grad():
            self.assertTrue(torch.equal(dit(**kwargs, memory_cond=c_mem), dit(**kwargs)))

    def test_invalid_slots_do_not_leak(self):
        """Empty slots are masked: their content cannot change the read."""
        for fusion in ("adaln_add", "prefix_tokens"):
            mem = make_memory("learned", fusion=fusion)
            if fusion == "adaln_add":
                torch.nn.init.normal_(mem.reader.out_proj[-1].weight)
            feats, age, mask = history(5)  # < K + W: some slots stay empty
            state, _ = mem.replay(feats, age, mask)
            ref = mem.read(state, torch.zeros(1))
            state.tokens[~state.valid] = 1e4
            state.time[~state.valid] = -1e4
            out = mem.read(state, torch.zeros(1))
            if fusion == "adaln_add":
                self.assertTrue(torch.allclose(out, ref, atol=1e-6), fusion)
                self.assertGreater(float(ref.abs().sum()), 0.0)
            else:
                (tok, tmask), (ref_tok, _) = out, ref
                self.assertEqual(int(tmask.sum()), 5 * 4)  # 5 frames x 2x2 read tokens
                self.assertTrue(torch.equal(tok, ref_tok), fusion)

    def test_oracle_follows_labels(self):
        n = 30
        feats, age, mask = history(n)
        keep = {3, 11, 17}
        u = torch.zeros(1, n, n)
        u[..., list(keep)] = 1.0
        state, _ = make_memory("oracle").replay(feats, age, mask, utility=u)
        self.assertTrue(keep.issubset(kept(state)))

    def test_rank_loss_trains_scorer_only(self):
        mem = make_memory("learned").train()
        n = 30
        feats, age, mask = history(n)
        _, loss = mem(feats, age, mask, utility=torch.rand(1, n, n))
        self.assertIsNotNone(loss)
        loss.backward()
        self.assertTrue(any(float(p.grad.abs().sum()) > 0 for p in mem.scorer.parameters() if p.grad is not None))
        self.assertTrue(all(p.grad is None for p in mem.reader.parameters()))

    def test_scorer_learns_oracle_ranking(self):
        """With labels that favour some frames, training makes the scorer keep them."""
        torch.manual_seed(0)
        mem = make_memory("learned").train()
        opt = torch.optim.AdamW(mem.scorer.parameters(), lr=3e-3)
        n = 30
        good = torch.zeros(n, dtype=torch.bool)
        good[::4] = True

        def sample():
            feats, age, mask = history(n, seed=int(torch.randint(0, 10**6, ())))
            feats[:, good, :, : D // 2] += 2.0  # frames worth keeping look different (LN-visible)
            return feats, age, mask, good.float().expand(1, n, n).clone()

        for _ in range(150):
            feats, age, mask, u = sample()
            _, loss = mem(feats, age, mask, utility=u)
            opt.zero_grad()
            loss.backward()
            opt.step()
        mem.eval()
        feats, age, mask, _ = sample()
        state, _ = mem.replay(feats, age, mask)
        queue = state.src[0, :K][state.valid[0, :K]]
        self.assertGreaterEqual(float(good[queue].float().mean()), 0.8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
