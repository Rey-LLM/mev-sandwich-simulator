"""Regenerate tests/vectors/uniswap_v3_pool_swaps.json from the Uniswap v3-core snapshot.

Usage:
    curl -sSLO https://raw.githubusercontent.com/Uniswap/v3-core/main/test/__snapshots__/UniswapV3Pool.swaps.spec.ts.snap
    python tools/convert_v3_swap_snapshot.py UniswapV3Pool.swaps.spec.ts.snap tests/vectors/uniswap_v3_pool_swaps.json

The pool and swap definitions below transcribe TEST_POOLS and
DEFAULT_POOL_SWAP_TESTS from v3-core test/UniswapV3Pool.swaps.spec.ts. The
expected values are taken verbatim from the snapshot; nothing is computed by
this project's AMM code.
"""
import json, re, sys
from decimal import Decimal, getcontext, ROUND_HALF_UP, ROUND_FLOOR
getcontext().prec = 200
SNAP = sys.argv[1]; OUT = sys.argv[2]
text = open(SNAP).read()
entries = {}
for m in re.finditer(r"exports\[`UniswapV3Pool swap tests (.*?) 1`\] = `\nObject \{\n(.*?)\n\}\n`;", text, re.S):
    body = {}
    for line in m.group(2).splitlines():
        k, v = line.strip().rstrip(',').split(': ', 1)
        body[k.strip('"')] = json.loads(v)
    entries[m.group(1)] = body
print(len(entries))

def bn40(x):
    return x.quantize(Decimal(1).scaleb(-40), rounding=ROUND_HALF_UP)
def encode_price_sqrt(r1, r0):
    q = bn40(Decimal(r1) / Decimal(r0))
    s = bn40(q.sqrt())
    return int((s * (Decimal(2) ** 96)).to_integral_value(rounding=ROUND_FLOOR))

E18 = 10**18
MIN_SQRT, MAX_SQRT = 4295128739, 1461446703485210103287273052203988822378723970342
import math
def min_tick(s): return math.ceil(-887272 / s) * s
def max_tick(s): return (887272 // s) * s
def max_liq(s): return (2**128 - 1) // ((max_tick(s) - min_tick(s)) // s + 1)
SP = {500: 10, 3000: 60, 10000: 200}
def full(fee, liq=2*E18):
    s = SP[fee]; return [[min_tick(s), max_tick(s), liq]]
M = 3000; S = 60
POOLS = [
 ("low fee, 1:1 price, 2e18 max range liquidity", 500, encode_price_sqrt(1,1), full(500)),
 ("medium fee, 1:1 price, 2e18 max range liquidity", M, encode_price_sqrt(1,1), full(M)),
 ("high fee, 1:1 price, 2e18 max range liquidity", 10000, encode_price_sqrt(1,1), full(10000)),
 ("medium fee, 10:1 price, 2e18 max range liquidity", M, encode_price_sqrt(10,1), full(M)),
 ("medium fee, 1:10 price, 2e18 max range liquidity", M, encode_price_sqrt(1,10), full(M)),
 ("medium fee, 1:1 price, 0 liquidity, all liquidity around current price", M, encode_price_sqrt(1,1),
   [[min_tick(S), -S, 2*E18], [S, max_tick(S), 2*E18]]),
 ("medium fee, 1:1 price, additional liquidity around current price", M, encode_price_sqrt(1,1),
   [[min_tick(S), max_tick(S), 2*E18], [min_tick(S), -S, 2*E18], [S, max_tick(S), 2*E18]]),
 ("low fee, large liquidity around current price (stable swap)", 500, encode_price_sqrt(1,1), [[-10, 10, 2*E18]]),
 ("medium fee, token0 liquidity only", M, encode_price_sqrt(1,1), [[0, 2000*S, 2*E18]]),
 ("medium fee, token1 liquidity only", M, encode_price_sqrt(1,1), [[-2000*S, 0, 2*E18]]),
 ("close to max price", M, encode_price_sqrt(2**127, 1), full(M)),
 ("close to min price", M, encode_price_sqrt(1, 2**127), full(M)),
 ("max full range liquidity at 1:1 price with default fee", M, encode_price_sqrt(1,1), full(M, max_liq(S))),
 ("initialized at the max ratio", M, MAX_SQRT - 1, full(M)),
 ("initialized at the min ratio", M, MIN_SQRT, full(M)),
]
INT256_MAX = 2**255 - 1
p50, p200, p52, p25 = encode_price_sqrt(50,100), encode_price_sqrt(200,100), encode_price_sqrt(5,2), encode_price_sqrt(2,5)
# (description, zero_for_one, amount_specified (+exact in / -exact out), limit or None)
CASES = [
 ("swap exactly 1.0000 token0 for token1", True, E18, None),
 ("swap exactly 1.0000 token1 for token0", False, E18, None),
 ("swap token0 for exactly 1.0000 token1", True, -E18, None),
 ("swap token1 for exactly 1.0000 token0", False, -E18, None),
 ("swap exactly 1.0000 token0 for token1 to price 0.50000", True, E18, p50),
 ("swap exactly 1.0000 token1 for token0 to price 2.0000", False, E18, p200),
 ("swap token0 for exactly 1.0000 token1 to price 0.50000", True, -E18, p50),
 ("swap token1 for exactly 1.0000 token0 to price 2.0000", False, -E18, p200),
 ("swap exactly 0.0000000000000010000 token0 for token1", True, 1000, None),
 ("swap exactly 0.0000000000000010000 token1 for token0", False, 1000, None),
 ("swap token0 for exactly 0.0000000000000010000 token1", True, -1000, None),
 ("swap token1 for exactly 0.0000000000000010000 token0", False, -1000, None),
 ("swap token1 for token0 to price 2.5000", False, INT256_MAX, p52),
 ("swap token0 for token1 to price 0.40000", True, INT256_MAX, p25),
 ("swap token0 for token1 to price 2.5000", True, INT256_MAX, p52),
 ("swap token1 for token0 to price 0.40000", False, INT256_MAX, p25),
]
out = {"source": "https://github.com/Uniswap/v3-core/blob/main/test/__snapshots__/UniswapV3Pool.swaps.spec.ts.snap",
       "spec": "https://github.com/Uniswap/v3-core/blob/main/test/UniswapV3Pool.swaps.spec.ts",
       "pools": []}
used = 0
for desc, fee, sp, positions in POOLS:
    pool = {"description": desc, "fee": fee, "tick_spacing": SP[fee], "sqrt_price_x96": str(sp),
            "positions": [[a, b, str(l)] for a, b, l in positions], "swaps": []}
    for cdesc, z, amt, lim in CASES:
        key = f"{desc} {cdesc}"
        exp = entries[key]; used += 1
        pool["swaps"].append({"description": cdesc, "zero_for_one": z, "amount_specified": str(amt),
            "sqrt_price_limit_x96": None if lim is None else str(lim), "expected": exp})
    out["pools"].append(pool)
assert used == len(entries), (used, len(entries))
json.dump(out, open(OUT, "w"), indent=1)
