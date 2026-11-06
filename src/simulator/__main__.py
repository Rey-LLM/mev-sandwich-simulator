"""Command line: ``python -m simulator --mode moderate|brutal``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from .block_builder import clamp_gas_limit
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
    parser.add_argument("--layouts", type=int, default=None, help="block layouts fuzzed per slot (brutal)")
    parser.add_argument("--gas-limit", type=int, default=None, help="block gas limit, at most 30000000")
    parser.add_argument("--trace", action="store_true", help="log every sealed block's tx trace to stderr")
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in {
        "seed": args.seed, "blocks": args.blocks, "txs_per_block": args.txs, "fuzz_iterations": args.fuzz,
        "layout_fuzz_iterations": args.layouts,
        "block_gas_limit": None if args.gas_limit is None else clamp_gas_limit(args.gas_limit),
    }.items() if v is not None}
    if args.trace:
        logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
        logging.getLogger("simulator.block_builder").setLevel(logging.DEBUG)
    cfg = SimulationConfig.for_mode(args.mode, workers=args.workers, **overrides)
    report = SimulationController(cfg).run()
    json.dump(report, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    result = report["result"]
    ok = result["invariants"]["ok"] and result.get("pbs", {}).get("invariants_ok", True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
