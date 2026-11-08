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
Layer 09: ``config`` gains the third run matrix, MAX Brutal Simulation, and
the CPU policies (capped = 50-60% of CPUs, full = every logical CPU);
``parallel`` sizes the pool by policy, grows it on demand, is safe to share
between threads and reports per-process telemetry and load probes;
``shard_density`` scales every fixed shard count; ``path_fuzzer`` runs
thousands of policy-gradient paths concurrently with per-step invariants.
Layer 10: ``validator`` (pre-flight integrity audit: pool state invariants,
TickMath / SqrtPriceMath / SwapMath rounding, signal bounds and an AST scan
of the simulator's own source; the controller refuses to simulate on a
blocking finding), ``config_sync`` (the central ``sandwich.toml``: one
schema for the file, the CLI flags and the GUI form; format-preserving,
locked, conflict-checked atomic writes) and ``settings_gui`` (browser
settings editor, ``python -m simulator --gui``).
Layer 11: ``ai_agent_factory`` (local embedded AI agent factory: a NumPy
neural matrix engine with manual backprop and AdamW, conditioned on the
Layer 08/09 RL results, synthesises competitor bot profiles - priority-fee
ranges, slippage thresholds, multi-hop path choices, target pools and bundle
layout - spins up a Python class per bot at runtime, deploys hundreds of
them per generation in parallel arena heats against the RL incumbent through
one block-building queue, and retrains on the winners with adaptive
mutation and exact capital / offspring allocation). It is the only module
that needs NumPy at import time.
Layer 12: ``pattern_ingestor`` (MEV pattern ingestion: normalisers for
Flashbots blocks-API exports, mev-inspect-py sandwich rows, security
advisories and normalised JSON / JSONL / CSV, a seeded synthetic reference
corpus; a local store of CRC-framed segments plus an SQLite index with a
500 GiB default cap, chunked streaming reads and a byte-budgeted LRU cache;
four category matrices; exact v2 slippage evaluation; parallel shard scans
on the worker pool). ``ai_agent_factory`` pre-trains its generator on the
stored patterns and seeds part of generation 0 from them. NumPy is imported
only for ``PatternStore.matrix``.

0.13.0 (correctness pass): the v3 swap loop now steps by tick-bitmap word,
reproduces the SqrtPriceMath overflow branch and keeps net-zero ticks
initialized, so it matches all 240 v3-core pool-swap scenarios to the wei
(``tests/test_reference_vectors.py``); mempool nonces, ordering, out-of-gas
and after-gas sandwich accounting follow the protocol rules; the builder has
an opt-in ``strict_balances`` funding mode. See README.md for what is
reference-checked, what is a model and what is experimental.

Layer 13 (0.14.0): ``attack_vectors`` (multi-victim sandwich, JIT liquidity
with exact v3 mint / burn rounding and fee growth, back-run cycle arbitrage,
liquidation back-run and spot-vs-TWAP oracle manipulation on a lending model)
and ``strategy_discovery`` (MAP-Elites search over primitive action genomes
that finds attack compositions outside the predefined templates, minimises
and labels them, and keeps them in a local, replay-verified JSONL library).

Layer 14 (0.15.0): ``custom_builder`` (custom bot architecture builder: a
validated JSON blueprint schema with nested trigger trees over victim
metrics, modelled opcode traces and event listeners; a nested action script
with transient storage, conditionals, unrolled loops, atomic rollback blocks
and delayed cross-chain legs; a compiler to forward-only bytecode and one
runtime class per blueprint; and a NumPy blueprint synthesizer driven by
``ai_agent_factory`` that runs on the MAX worker pool).

Layer 15 (0.16.0): ``code_factory`` (writes generated bots to disk as
Python modules and Solidity parameter records, deploys them from disk -
digest-checked and AST allow-listed - in arena heats, retrains the generator
and rewrites the files), ``tx_graph`` (versioned BTC / ETH dataflow DAGs with
exact flows, competition and replacement anomalies, layered layout and an
interactive HTML view), ``gui`` (telemetry window and HTML dashboard: per-core
CPU load, synthesis speed, convergence) and ``i18n`` (Russian by default).

Layer 16 (0.17.0): ``sandwich_detector`` (finds sandwiches in real Ethereum
blocks from receipts / logs or call traces, over JSON-RPC or from a saved
export: attacker, victims, each victim's counterfactual loss, attacker profit
after gas and coinbase bribes; checked on mainnet block 12775690) and
``sandwich_guard`` (pre-trade check: is a swap sandwichable at a given
slippage, the largest safe slippage and the largest safe size).
"""

__version__ = "0.17.0"
