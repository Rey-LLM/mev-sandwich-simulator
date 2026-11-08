# MEV Sandwich Simulator

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Uniswap v3: bit-exact](https://img.shields.io/badge/Uniswap%20v3-bit--exact%20(240%2F240%20vectors)-ff007a)](tests/test_reference_vectors.py)
[![Real mainnet block](https://img.shields.io/badge/detector-checked%20on%20mainnet%20block%2012775690-627eea)](tests/test_sandwich_detector.py)
[![Dependencies](https://img.shields.io/badge/runtime%20deps-numpy%20only-informational)](pyproject.toml)

**Find sandwich attacks in real Ethereum blocks, measure what every victim lost
to the wei, check a swap before you send it, and simulate the whole MEV
supply chain offline.** Pure Python, exact integer math, Uniswap v2 / v3 / v4.

* **Sandwich detector for mainnet blocks.** Point it at an RPC node or a saved
  block export. It reports the attacker (EOA and bot contract), every victim,
  each victim's loss as a counterfactual "what you would have got without the
  front-run", and the attacker's gross and net profit after gas and coinbase
  bribes.
* **Pre-trade check.** Is my swap sandwichable at the slippage I am about to
  sign? What slippage is safe? How small must the trade be?
* **Reference AMM math.** The Uniswap v3 swap loop reproduces all 240 official
  `v3-core` pool scenarios to the wei. The detector and the checker run on the
  same verified code.
* **An offline lab** with a seeded mempool, a PBS / MEV-Boost auction, a
  Flashbots Protect model, multi-hop routing, JIT liquidity, oracle attacks and
  strategy search, all deterministic and reproducible.

It is a **measurement and teaching tool, not a trading bot.** It holds no keys
and never signs or sends a transaction. The detector and the pre-trade check
only *read* chain data; everything else runs on markets generated in memory
from a seed.

## Contents

* [Real-block sandwich detector](#real-block-sandwich-detector)
* [Pre-trade check: is my swap sandwichable?](#pre-trade-check-is-my-swap-sandwichable)
* [What you can trust, and why](#what-you-can-trust-and-why)
* [What the simulator does](#what-the-simulator-does)
* [Quick start](#quick-start)
* [Layout](#layout)

## Real-block sandwich detector

```bash
export ETH_RPC_URL=https://your-node.example          # any Ethereum JSON-RPC endpoint
PYTHONPATH=src python -m simulator --detect-block latest
PYTHONPATH=src python -m simulator --detect-block 19000000 --block-count 10 --save-export blocks.json.gz
PYTHONPATH=src python -m simulator --detect-file blocks.json.gz     # offline, from the saved export
PYTHONPATH=src python -m simulator --detect-file tests/vectors/mainnet_block_12775690_traces.json.gz
```

The last command runs on a real block that ships with the tests. Block
12775690 (6 July 2021) holds a sandwich on the ShibaSwap BONE/WETH pair:

| | Transaction | What happened |
|---|---|---|
| Front-run | tx 2 `0x91a3ab…17a0` | bot `0x000000000027d2ef…c4d8` buys BONE for 12.1088 WETH |
| Victim 1 | tx 3 `0x9b40de…5934` | buys exactly 60 BONE, **overpays 0.00258 WETH** |
| Victim 2 | tx 4 `0xf8e45a…15f9` | 0.3 WETH in, **gets 0.1094 BONE less** (0.00119 WETH) |
| Victim 3 | tx 5 `0xdf63b2…f00b` | 0.125 WETH in, **gets 0.0456 BONE less** (0.00049 WETH) |
| Victim 4 | tx 6 `0x1fe35f…4f4f` | 30 WETH in, **gets 10.855 BONE less** (0.11827 WETH) |
| Back-run | tx 7 `0xc300d1…95a2` | sells the same BONE for 12.1588 WETH |

Attacker: **+0.049991 WETH gross**, minus 0.0317 ETH paid straight to the miner
(gas price was 1 wei, a Flashbots-era bundle), **+0.018271 ETH net**. Victims
together: **−0.12253 ETH**, about 39.5 bps each. All four victim losses are
marked `exact`: the same AMM code, started from the same reserves,
reproduces each victim's real on-chain output to the wei. The test suite also
checks the attacker's profit against the raw WETH transfers in the traces and
recomputes a victim by hand from the `getReserves()` reads in the block.

**How it works.** The detector reads either receipts with logs (`Swap`,
`Sync`; what any node returns) or call traces (Parity `trace_block`, the
format `flashbots/mev-inspect-py` uses). Per pool and in block order it looks
for a front-run A→B, then victims trading A→B from other senders, then a
back-run B→A from the same EOA or the same bot contract (never a public
router) that sells 50–110% of what the front-run bought. For each victim it
replays the pool *without* the front-run:

* **v2:** reserves before the front-run come from `Sync` events or
  `getReserves()` outputs and are carried through every swap by its exact
  deltas. The pair's fee is calibrated from its own swaps (0.3% Uniswap, 0.25%
  and other forks). A victim that fixed its input loses output (`getAmountOut`);
  a victim that fixed its output overpays input (`getAmountIn`). Swaps in
  between are replayed in the mode they were sent in.
* **v3:** a log only gives the price *after* a swap. The price before the
  front-run is recovered by bisection over the verified v3 swap loop: the
  start price whose forward run lands exactly on the logged `sqrtPriceX96` and
  output. If no price reproduces it, the swap crossed a tick, and the loss is
  either reported as approximate or not reported.

**Limits, stated plainly.** Bribes paid through internal calls are only
visible in trace data. v3 pools in trace data carry no price, so only
detection and profit are reported there. Fee-on-transfer tokens break the
trace-based input amounts (those victims are not marked exact). The matching
is a heuristic, like every public sandwich classifier: a bot that splits its
legs across unrelated addresses will not be linked.

## Pre-trade check: is my swap sandwichable?

```bash
# a swap described in JSON (pool state, token_in, amount_in, slippage_bps, gas_price_wei)
PYTHONPATH=src python -m simulator --check-swap swap.json
# or read a live pool over RPC
PYTHONPATH=src python -m simulator --check-pool 0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc \
    --token-in 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2 --amount-in 10000000000000000000 \
    --slippage-bps 100 --gas-price-wei 20000000000 --lang en
```

The attacker's best front-run is searched exactly: the largest front-run the
victim's `amountOutMin` still tolerates is found by bisection, and the profit
curve below it is scanned and refined. The output says whether the swap is
sandwichable, the attacker's best net profit and your loss, the **largest
safe slippage** (attacker profit can only grow with tolerance, so it is a
bisection over basis points), and when needed the **largest safe trade size**
at your tolerance. Advice text is Russian by default, `--lang en` for English.

Two rules of thumb that fall out of the math (and are tested):

* A swap that moves the pool by less than twice the pool fee is never
  sandwichable, at any slippage: the attacker pays the fee twice.
* Above that threshold, gas is what protects you: with free gas even a 1 bps
  tolerance pays the attacker (a 6 WETH swap into a 1,000 WETH pool).

## What you can trust, and why

The project separates code that is checked against an external reference from
code that is a model or an experiment. The table says which is which.

| Part | Module | Status | Checked against |
|---|---|---|---|
| Uniswap v2 `getAmountOut` / `getAmountIn` / pair `K` check | `slippage_engine` | **bit-exact** | `Uniswap/v2-core` `UniswapV2Pair.spec.ts` swap and flash-swap cases (output accepted, output + 1 wei rejected), `Uniswap/v2-periphery` router cases |
| Uniswap v3 TickMath, SqrtPriceMath, SwapMath | `slippage_engine` | **bit-exact** | every numeric vector in `Uniswap/v3-core` `TickMath.spec.ts` (+ snapshot), `SqrtPriceMath.spec.ts`, `SwapMath.spec.ts` |
| Uniswap v3 full pool swaps | `slippage_engine.ConcentratedPool` | **bit-exact** | all **240** scenarios of `v3-core` `UniswapV3Pool.swaps.spec.ts.snap` (15 pools x 16 swaps): both token deltas, final tick, `feeGrowthGlobalX128` and reverts, to the wei |
| EIP-1559 base-fee update, effective tip, fee charging | `block_builder` | **exact rules** | EIP-1559 formula; geth `consensus/misc/eip1559` vectors |
| Nonce rules, price-and-nonce ordering, tx replacement (10% bump) | `block_builder`, `mempool_emulator` | **exact rules** | geth `TransactionsByPriceAndNonce` / legacypool behaviour (integer threshold) |
| Sandwich P&L in tokens | `mempool_emulator`, `block_builder` | **exact** for the simulated state | every leg is executed with the exact AMM math above |
| Gas usage | `mempool_emulator.estimate_swap_gas` | **estimate** | per-hop constants plus per-tick-cross cost; not EVM-metered |
| Sandwich P&L in ETH / after gas | `block_builder`, `mempool_emulator` | **estimate** | uses the gas estimate above and converts tokens at the spot price of a direct pool against WETH |
| Uniswap v4 | `slippage_engine.V4Pool` | **model** | v4 swap math equals v3 (verified above) plus dynamic LP fee, protocol fee (max 1000 pips) and the 100% fee rule; hooks other than fee overrides are not modelled |
| PBS / MEV-Boost auction, relays, proposer | `block_builder` | **model** | follows the public flow; relay checks re-execute blocks in this simulator, not on an EVM |
| Flashbots Protect / MEV-Share | `private_relay` | **model** | built from public docs; hint inference is a heuristic; signatures and envelopes are simulation-only crypto (SHA3/HMAC/SHA-256 keystream, not keccak or real encryption) |
| Multi-hop router, cycle search | `multi_hop_router` | **exact path math**, heuristic search | Bellman-Ford cross-check on log prices |
| RL bidding agent, policy-path fuzzer | `reinforcement_engine`, `path_fuzzer` | **experimental** | toy agents on a synthetic market; results say nothing about mainnet |
| "AI agent factory" | `ai_agent_factory` | **experimental** | parameter vectors from a small NumPy MLP, scored in simulated auctions |
| Pattern store / normalisers | `pattern_ingestor` | **experimental** | only exercised on the built-in synthetic corpus (`source="synthetic"`) |
| Attack vectors: multi-victim sandwich, JIT liquidity, back-run arbitrage | `attack_vectors` | **exact** token amounts | every leg runs on the AMM math above; v3 mint rounds up and burn rounds down as in `_modifyPosition`; JIT fees are split at the position's ticks and checked against the single-swap result on every swap |
| Attack vectors: liquidation back-run, oracle manipulation | `attack_vectors` | **model** | Aave-style lending market (LTV, threshold, close factor, bonus), spot or per-block TWAP oracle; swap legs are exact |
| Strategy discovery | `strategy_discovery` | **experimental** search, **exact** replay | MAP-Elites search over action sequences; every saved strategy replays to the same wei |
| Custom bot builder: compiler, VM, swap / victim / mint / burn / cycle legs | `custom_builder` | **exact** token amounts | legs run on `AttackState`; a compiled sandwich equals the hand-written one to the wei (`tests/test_custom_builder.py`) |
| Custom bot builder: opcode traces, `bridge` legs, AI blueprint synthesis | `custom_builder` | **model** / **experimental** | opcode traces are a fixed pattern per pool version, not an EVM trace; the remote chain is another seeded scenario |
| Code factory: generated `.py` bots, disk deployment, retraining | `code_factory` | **experimental** | arena P&L is exact wei; every file is digest-checked and AST allow-listed before it runs; `.sol` files are parameter records checked by the factory's own validator (no `solc` offline) |
| ETH transaction graph | `tx_graph` | **exact** for the simulated block | every included tx is replayed on the parent state; edges must match the replayed balance and reserve deltas to the wei and the replayed state root must equal the sealed block's |
| BTC transaction graph | `tx_graph` | **model** | synthetic seeded UTXO mempool; P2WPKH size estimate, BIP 125 rules 1/3/4 plus fee rate, ancestor-feerate block template; no real Bitcoin data |
| Sandwich detector: profit, victim loss (v2) | `sandwich_detector` | **exact** | real mainnet block 12775690: attacker profit equals the raw WETH transfers in the traces; every victim's real output is reproduced to the wei before its counterfactual is trusted |
| Sandwich detector: victim loss (v3, from logs) | `sandwich_detector` | **exact** when a victim stays inside one tick range, else **approximate** | the reconstructed pre-front-run price must reproduce the front-run's logged price and output exactly; synthetic node-format blocks in `tests/test_sandwich_detector.py` |
| Sandwich matching (who attacked whom) | `sandwich_detector` | **heuristic** | same pool, A→B / A→B / B→A, front and back share an EOA or a non-router contract, back-run sells 50-110% of the front-run |
| Pre-trade check | `sandwich_guard` | **exact** for the pool state given | brute-force scans in `tests/test_sandwich_guard.py`; safe slippage and safe size are checked at the boundary (safe, safe + 1) |
| Telemetry window / HTML dashboard | `gui` | tooling | `/proc/stat` deltas per core (load average fallback); charts are plain plans checked in tests |

### How the v3 math is verified

`tests/test_reference_vectors.py` replays the official Uniswap test vectors.
The 240 full-pool scenarios come from a real Hardhat run of the v3-core
contracts; `tests/vectors/uniswap_v3_pool_swaps.json` is a mechanical
conversion of that snapshot (`tools/convert_v3_swap_snapshot.py` regenerates
it and documents where every pool and swap definition comes from). The test
builds each pool from the same positions, runs the same swap and requires
equality to the wei.

Three details make the port bit-exact rather than "close":

1. **Tick-bitmap stepping.** The contract ends a swap step at the edge of a
   256-tick bitmap word even when no tick there is initialized. Splitting a
   swap into the same steps changes rounding by a few wei, and in one
   scenario changes the final tick. `ConcentratedPool` emulates
   `TickBitmap.nextInitializedTickWithinOneWord` exactly.
2. **The uint256 overflow branch** of
   `SqrtPriceMath.getNextSqrtPriceFromAmount0RoundingUp`. Python integers do
   not overflow, so the fallback formula the contract uses for huge inputs is
   reproduced explicitly. Without it, the "close to max price" scenarios end
   hundreds of ticks away from the contract's result.
3. **liquidityGross.** A tick stays initialized while its gross liquidity is
   positive, even if its net liquidity is 0, and the contract still stops a
   step there.

## What the simulator does

* **Moderate Scan** (`--mode moderate`): generates a seeded mempool of swaps,
  quotes each one exactly, reports price impact, slippage slack, gas
  under-provisioning, and fuzzes pool invariants. Nothing is mutated.
* **Brutal Aggressive Simulation** (`--mode brutal`): for every pending swap,
  searches the most profitable front-run size, then builds blocks on a forked
  state: front-run, victim, back-run, each executed with exact math and kept
  only if the sandwich still pays after gas. On top of that it runs the PBS
  model (builders, bundles, relays, proposer, EIP-1559 base fee), the private
  relay model, the router and the RL agent.
* **MAX Brutal Simulation** (`--mode max`): the same with every core and the
  experimental modules (policy-path fuzzer, AI agent factory, custom bot
  builder) turned up.

Rules applied in block execution:

* A transaction is invalid when `max_fee_per_gas < base_fee`, when
  `max_priority_fee_per_gas > max_fee_per_gas`, when its nonce is not the
  sender's next nonce, or when its gas limit does not fit the block. Invalid
  transactions are not included and leave state untouched.
* An included transaction pays `gas_used * (base_fee + effective_tip)`; the
  base fee is burned and the tip goes to the block's coinbase. A reverted or
  out-of-gas transaction still pays for gas and uses its nonce.
* Bundles are atomic: if any transaction fails (outside
  `reverting_tx_hashes`) the whole bundle is dropped and state is restored.

## Attack types and local strategy discovery

`attack_vectors` adds attack types beyond the plain sandwich. Each is a
sequence of primitive actions (`swap`, `victim`, `mint`, `burn`, `liquidate`,
`deposit`, `borrow`, `settle`) executed on a private copy of a seeded
scenario: two WETH/USDC venues (v2 and v3), WETH/DAI (v4), DAI/USDC (v2) and
a WETH-collateral lending market whose oracle is the v2 pool.

| Type | What it does |
|---|---|
| `sandwich` | front-run, victim, back-run |
| `multi_victim_sandwich` | one front-run ahead of every same-direction victim that still clears its limit, one back-run |
| `jit_liquidity` | mint concentrated liquidity right before a large v3/v4 swap, burn and collect right after (victims get a better price) |
| `backrun_arbitrage` | after a victim moves a pool, close the gap through a 2- or 3-pool cycle back to WETH |
| `liquidation_backrun` | a victim moves the oracle pool, a loan becomes unhealthy, liquidate it and sell the collateral |
| `oracle_manipulation` | push a spot-price oracle, borrow against inflated collateral; also run against a TWAP oracle, which removes the profit |

Every result reports token deltas, an estimated gas cost, net profit in WETH
(exact when the result is `flat`, i.e. every other token nets to zero) and
each victim's lost output against a run without the attacker.

`strategy_discovery` searches for compositions nobody defined. A candidate is
a short genome of those actions (sizes as integers, pools by index, victims by
role: pool, direction, size rank). The engine:

1. seeds generation 0 with the predefined attacks written as genomes plus
   random genomes, on several scenario seeds;
2. mutates (resize, flip direction, insert, delete, reorder, duplicate) and
   crosses over the elites of a MAP-Elites archive keyed by the executed
   *structure signature*, e.g. `MINT[V1] SWAP[V1=] VICTIM[V1] BURN[V1] SETTLE[V1~]`,
   so structurally different strategies survive side by side;
3. scores each candidate by exact execution, after settling every balance to
   WETH and subtracting the gas estimate;
4. drops signatures that match a predefined template, minimises the rest
   (removes every action that does not add profit), re-runs them on hold-out
   seeds, labels them from their components (for example
   "JIT liquidity on V1 + front-run on V1 + back-run on V1") and keeps the best
   record per label family;
5. appends them to a local JSON Lines library, by default
   `~/.mev_sim/strategies/strategies.jsonl` (`$MEV_SIM_STRATEGY_DIR` or
   `--strategy-dir` to change it). `--verify-strategies` replays every record
   and requires the stored profit to the wei.

```bash
PYTHONPATH=src python -m simulator --attack-catalog --scenario-victims 10
PYTHONPATH=src python -m simulator --discover --generations 60 --population 64
PYTHONPATH=src python -m simulator --strategies
PYTHONPATH=src python -m simulator --verify-strategies
```

A discovered strategy says that an action order pays *in this model*. Typical
finds are hybrids such as JIT + sandwich on the same pool, cross-pool
sandwiches that front-run on one victim's pool and back-run on another, and
"push the oracle pool, then liquidate". The last one only works because the
lending market reads a spot price; the same scenario with a TWAP oracle does
not pay. Hold-out results show how often a structure also pays on other seeds.

## Custom bot builder

`custom_builder` turns a JSON *blueprint* into a runnable competitor bot. A
blueprint has four parts:

* `trigger`: a nested tree (`all`, `any`, `not`) over victim metrics (priority
  fee, gas limit, size in ppm of the reserve, slippage tolerance, price impact,
  ticks crossed, cross-venue spread, loan health after the victim, ...),
  counts or ordered sequences in a modelled opcode trace, and event listeners
  (`Swap`, `LargeSwap`, `PriceDeviation`, `LoanAtRisk`, `TickCross`);
* `script`: nested actions (`frontrun`, `victim`, `victims`, `backrun`, `swap`,
  `close`, `mint`, `burn`, `liquidate`, `arb`, `tstore` / `tswap` transient
  storage, `if`, `repeat`, `atomic` with rollback or abort, and `bridge`, a
  delayed leg on a second seeded "chain");
* `tracing`: opcode conditions, contiguous matching, trace recording;
* `pipeline`: revert policy, settlement, gas and capital caps, the optimiser
  budget. Every key can be overridden per run without recompiling.

The compiler validates the blueprint (every problem is reported with its
path, e.g. `$.script[2].size.ppm`), assembles it into bytecode for a small
stack / jump machine whose jumps only go forward (so every bot terminates),
and creates one Python class per blueprint, cached by digest and safe to share
between threads. One `{"optimize": [lo, hi]}` size per blueprint is searched
per victim with the same grid + ternary search the attack catalog uses.

```bash
PYTHONPATH=src python -m simulator --bot-templates
PYTHONPATH=src python -m simulator --bot-template composite_bridge_bundle --seed 3
PYTHONPATH=src python -m simulator --build-bot my_bot.json --scenario-victims 10
PYTHONPATH=src python -m simulator --blueprint-schema
```

In MAX mode (`custom_bots`, `custom_generations` in `sandwich.toml`) the AI
agent factory's NumPy generator, conditioned on the RL results, designs whole
blueprints, trigger structure included. They are compiled and scored on
seeded scenarios in fixed batches on the full-core worker pool, the generator
retrains on the best ones, and every trigger structure that no built-in
template uses is listed under `custom_builder.novel_trigger_classes`. "Novel"
means new to this design space, not a claim about real searchers.

## Layer 15: code factory, transaction graph, telemetry, Russian UI

**Code factory** (`code_factory`). The Layer 11 NumPy generator produces bot
profiles; the factory writes each one to disk as an independent Python module
(`bot_NNNN.py`: literal constants, one `SynthesizedBot` subclass, `build()`)
and a Solidity parameter record (`bot_NNNN.sol`: a `library` of `uint256`
constants, no functions). Writes are atomic and listed with SHA-256 digests in
`manifest.json`. Arena heats then load the bots **from disk** in the worker
processes (digest check, AST allow-list, compile), the Layer 04 builder decides
who lands, the generator retrains on the elite and the files are rewritten for
the next generation; the best bot of each generation is kept in `champions/`.
`block_builder.generated_competitors()` / `run_generated_competition()` put the
generated bots into a builder competition. Default directory:
`src/simulator/generated_bots/` (not tracked by git), or `--bot-dir` /
`$MEV_SIM_BOT_DIR`.

**Transaction graph** (`tx_graph`). A versioned dataflow DAG: an account or
pool gets a new node each time a transaction changes it, so the graph stays
acyclic. ETH graphs replay a sealed block and flag value-conservation breaks,
state-root mismatches, sandwiches (with the attacker's exact token gain),
builder competition per victim (winning margin in wei, "photo finish" under a
threshold) and evicted bundles. BTC graphs are built from a synthetic UTXO
mempool and flag double spends, BIP 125 replacements, negative fees, unknown
inputs, dust and CPFP packages, and build an ancestor-feerate block template.
Layout: Kahn topological order, longest-path layers, barycentric crossing
reduction. Output: one self-contained HTML page (zoom, pan, click to highlight
upstream / downstream flow, exact amounts in tooltips).

**Telemetry** (`gui`). `--monitor` opens a dark tkinter window (CustomTkinter
if installed) with per-core CPU load, code synthesis speed and generator
convergence, plus buttons for the factory, a MAX Brutal CPU stress on every
core and the graph. `--monitor-html FILE` records the same while the factory
runs and writes an HTML dashboard (no display needed).

**Language.** Human-facing text is Russian by default: `sandwich.toml`
comments, the settings editor, the telemetry window, HTML pages, factory logs
and file headers, CLI help and error prefixes (`[ui] language`, `--lang en`
or `$MEV_SIM_LANG` switch to English). JSON keys, TOML keys and flag names
stay English so scripts keep working; argparse's own usage/error wording also
stays English.

```bash
PYTHONPATH=src python -m simulator --code-factory --factory-bots 32 --factory-generations 4
PYTHONPATH=src python -m simulator --tx-graph graph.html
PYTHONPATH=src python -m simulator --monitor-html telemetry.html --stress-seconds 5
PYTHONPATH=src python -m simulator --monitor            # needs tkinter and a display
```

## What it does not do

* No live data in the simulator itself: no mempool subscription, no
  historical replay. Only the detector and the pre-trade check read chain
  data (read-only JSON-RPC calls, or a saved export).
* No EVM: gas is estimated, not metered; storage, calldata and contract code
  are not executed.
* No transaction signing, sending or key handling.
* No LP fee accrual to pre-existing positions (only the attacker's own JIT
  positions track fees), no v3 protocol fee (`feeProtocol` is 0), no
  on-chain TWAP oracle (the lending model's TWAP is a mean of per-block spot prices), no v4 hooks beyond fee overrides.
* By default accounts are treated as funded: balances are signed profit and
  loss ledgers and may go negative. Pass `strict_balances=True` to
  `BuilderState` to enforce the funding rules (insufficient funds for
  `gas_limit * max_fee + value` makes a transaction invalid, spending more
  tokens than held reverts, an unpayable coinbase transfer drops the bundle).
* The synthetic mempool is deliberately aggressive (trades up to 3% of a
  pool's reserve, slippage tolerances up to 10%), so the extracted amounts
  are not estimates of real-world MEV.

## Quick start

```bash
pip install -e '.[test]'           # Python 3.10+, numpy is the only runtime dependency
python -m pytest -q                # full suite, includes the reference vectors
python -m pytest -q tests/test_reference_vectors.py   # only the Uniswap vectors

PYTHONPATH=src python -m simulator --detect-file tests/vectors/mainnet_block_12775690_traces.json.gz
PYTHONPATH=src python -m simulator --mode moderate     # quick scan, JSON report on stdout
PYTHONPATH=src python -m simulator --mode brutal       # sandwiches, PBS, relay, router, RL
PYTHONPATH=src python -m simulator --audit-only        # pre-flight integrity audit only
PYTHONPATH=src python -m simulator --help              # every setting
```

Settings live in `sandwich.toml`; precedence is mode preset, then the file,
then `$MEV_SIM_WORKERS`, then `--set table.key=value`, then flags.
`python -m simulator --gui` opens a local settings editor on 127.0.0.1.

### Using the math directly

```python
from simulator.slippage_engine import ConcentratedPool, Position, V2Pool, evaluate_trade

pool = ConcentratedPool.from_positions(
    "USDC", "WETH", sqrt_price_x96=2**96,
    positions=[Position(-887220, 887220, 2 * 10**18)], fee_pips=3000, tick_spacing=60)
res = pool.quote_exact_in(10**18, zero_for_one=True)
print(res.amount_out, res.tick_after, res.fee_growth_global_x128_delta)

v2 = V2Pool("WETH", "USDC", 4_000 * 10**18, 12_000_000 * 10**6)
print(evaluate_trade(v2, "WETH", 10**18, tolerance_bps=50).to_dict())
```

## Layout

```
src/simulator/
  slippage_engine.py    AMM math (v2, v3, v4), slippage reports           [reference-checked]
  state_fork.py         snapshot / revert / fork of pools and balances
  mempool_emulator.py   seeded mempool, sandwich search, simplified blocks
  block_builder.py      EIP-1559, nonces, tx pool, bundles, PBS auction
  private_relay.py      Protect / MEV-Share model                         [model]
  multi_hop_router.py   token graph, routes, cycles
  invariants.py         property fuzzing of pool math
  validator.py          pre-flight audit run before every simulation
  config.py, config_sync.py, settings_gui.py, parallel.py, dataflow.py, controller.py
  attack_vectors.py     JIT, multi-victim, back-run arbitrage, liquidation, oracle attacks
  strategy_discovery.py local search for new attack compositions, strategy library      [experimental]
  custom_builder.py     blueprint schema, compiler, bytecode VM, AI blueprint synthesis   [exact legs, model traces]
  reinforcement_engine.py, path_fuzzer.py, ai_agent_factory.py, pattern_ingestor.py   [experimental]
  code_factory.py       writes bots to disk (.py / .sol), deploys them from disk, retrains  [experimental]
  tx_graph.py           BTC / ETH dataflow graphs, anomalies, interactive HTML    [ETH exact, BTC model]
  gui.py, i18n.py       telemetry window / dashboard, Russian and English text
  sandwich_detector.py  sandwiches in real blocks: logs or traces, RPC client, keccak   [exact v2, heuristic matching]
  sandwich_guard.py     pre-trade check: sandwichable?, safe slippage, safe size
  generated_bots/       default output of the code factory (git-ignored)
tests/
  test_reference_vectors.py   Uniswap v2/v3 official vectors (335 test cases)
  test_protocol_rules.py      EIP-1559, nonces, out-of-gas, funding, replacement, v4 fees
  test_custom_builder.py      blueprint schema, VM semantics, exactness, thread contention
  test_code_factory.py        generated-code allow-list, tamper rejection, disk deployment, determinism
  test_tx_graph.py            topology and layout math, exact ETH replay, BIP 125, block template, HTML
  test_gui_telemetry.py       CPU sampling math, thread-safe series, chart plans, dashboard
  test_i18n.py                catalog completeness, Russian sandwich.toml / editor / CLI
  test_layer15_stress.py      determinism with every core saturated
  test_sandwich_detector.py   real mainnet block, node-format synthetic blocks, negatives, RPC path
  test_sandwich_guard.py      best attack vs dense scans, safe slippage / size boundaries
  vectors/                    converted v3-core swap snapshot, mainnet block 12775690 (traces)
tools/convert_v3_swap_snapshot.py, tools/fetch_mainnet_fixture.py
```

## License and use

For research and education. Sandwich attacks harm the users they target; this
code exists to measure and explain that harm, not to perform it.
