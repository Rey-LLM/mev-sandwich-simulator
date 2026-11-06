"""Command line: ``python -m simulator --mode moderate|brutal``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .config import SimulationConfig
from .controller import SimulationController


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="simulator", description="Offline MEV sandwich simulator")
    parser.add_argument("--mode", default="moderate", help="moderate | brutal")
    parser.add_argument("--workers", type=int, default=None, help="clamped to 50-60%% of CPUs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--txs", type=int, default=None, help="pending txs per block")
    parser.add_argument("--fuzz", type=int, default=None, help="invariant fuzz iterations")
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in {
        "seed": args.seed, "blocks": args.blocks, "txs_per_block": args.txs, "fuzz_iterations": args.fuzz,
    }.items() if v is not None}
    cfg = SimulationConfig.for_mode(args.mode, workers=args.workers, **overrides)
    report = SimulationController(cfg).run()
    json.dump(report, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0 if report["result"]["invariants"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
