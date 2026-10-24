"""Write train/val episode-index files from inclusive episode ranges.

    python scripts/make_episode_splits.py --ranges 700-799 1500-1599 \
        --val_per_range 10 --out_prefix splits/robomme_small

writes ``splits/robomme_small_train.txt`` and ``splits/robomme_small_val.txt``.
Validation episodes are drawn per range (i.e. per task) with a fixed seed.
"""

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranges", nargs="+", required=True, help="inclusive START-END")
    parser.add_argument("--val_per_range", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train, val = [], []
    for spec in args.ranges:
        start, end = (int(v) for v in spec.split("-"))
        episodes = list(range(start, end + 1))
        held_out = sorted(rng.sample(episodes, args.val_per_range))
        val += held_out
        train += [ep for ep in episodes if ep not in set(held_out)]

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, eps in (("train", train), ("val", val)):
        path = prefix.parent / f"{prefix.name}_{name}.txt"
        path.write_text(" ".join(str(ep) for ep in sorted(eps)) + "\n")
        print(f"{path}: {len(eps)} episodes")


if __name__ == "__main__":
    main()
