"""MEV sandwich simulator core package.

Layer 02: ``slippage_engine`` (v2 / v3 / v4 AMM math, slippage vectors).
Layer 03: ``config`` (run modes, CPU worker ceiling), ``parallel`` (capped
multiprocessing pool), ``state_fork`` (snapshot / revert / fork),
``dataflow`` (token flow graph), ``invariants`` (XY=K and CL property checks),
``mempool_emulator`` (scan and multi-block sandwich search) and
``controller`` (entry point; also ``python -m simulator``).
Layer 04: ``block_builder`` (PBS / MEV-Boost auction, EIP-1559 + nonce
ordering, bundle simulation, sandwich and JIT guards, gas-capped sealing with
per-tx traces, parallel block layout fuzzer).
Layer 05: ``private_relay`` (Flashbots Protect / MEV-Share relay emulator:
private pool, hint leakage matrix, searcher inference, backrun-only bundle
auction with refunds, encrypted builder delivery, exposure analysis).
Layer 06: ``multi_hop_router`` (token graph, path and cycle enumeration,
Bellman-Ford cross-check, sequential v2 / v3 / v4 execution with combined
slippage degradation vectors, order splitting, parallel cycle scans).
Layer 08: ``reinforcement_engine`` (online Q-learning and policy-gradient
bidding agents trained against competing searcher bots, exact-wei rewards,
parallel policy iteration on the capped worker pool); ``block_builder`` gains
competitor bots and the exact bundle eviction model.
"""

__version__ = "0.8.0"
