"""MEV sandwich simulator core package.

Layer 02: ``slippage_engine`` (v2 / v3 / v4 AMM math, slippage vectors).
Layer 03: ``config`` (run modes, CPU worker ceiling), ``parallel`` (capped
multiprocessing pool), ``state_fork`` (snapshot / revert / fork),
``dataflow`` (token flow graph), ``invariants`` (XY=K and CL property checks),
``mempool_emulator`` (scan and multi-block sandwich search) and
``controller`` (entry point; also ``python -m simulator``).
"""

__version__ = "0.3.0"
