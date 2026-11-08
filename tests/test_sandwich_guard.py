"""Layer 16: pre-trade sandwich check (sandwichable or not, safe slippage, safe size)."""

import json
from fractions import Fraction

import pytest

from simulator.__main__ import main as cli_main
from simulator.sandwich_detector import WETH
from simulator.sandwich_guard import assess_swap, best_attack, pool_from_spec, simulate_attack
from simulator.slippage_engine import ConcentratedPool, MASEError, V2Pool, get_sqrt_ratio_at_tick, min_amount_out

TOKEN = "0x1111111111111111111111111111111111111111"
E18 = 10 ** 18


def v2_pool():
    return V2Pool(TOKEN, WETH, 2_000_000 * E18, 1_000 * E18, pool_id="pair")


def _profitable(pool, amount, slippage, gas=0, zfo=False):
    quoted = pool.quote_exact_in(amount, zfo).amount_out
    a = best_attack(pool, zfo, amount, min_amount_out(quoted, slippage), gas)
    return a is not None and a.net_profit > 0


def test_large_swap_with_loose_slippage_is_sandwichable():
    pool = v2_pool()
    r = assess_swap(pool, WETH, 10 * E18, slippage_bps=100, gas_price_wei=20 * 10 ** 9)
    assert r.sandwichable and r.gas_counted
    assert r.attack.victim_out >= r.min_out  # the attack never makes the victim revert
    assert r.victim_loss == r.quoted_out - r.attack.victim_out > 0
    assert r.attack.net_profit == r.attack.back_out - r.attack.front_in - r.gas_cost > 0
    assert pool.reserve0 == 2_000_000 * E18  # the caller's pool is untouched


def test_attack_is_replayed_exactly():
    pool = v2_pool()
    r = assess_swap(pool, WETH, 10 * E18, slippage_bps=100)
    assert simulate_attack(pool, False, r.attack.front_in, 10 * E18) == (
        r.attack.front_out, r.attack.victim_out, r.attack.back_out)


def test_best_attack_beats_a_dense_scan():
    pool = v2_pool()
    victim = 10 * E18
    min_out = min_amount_out(pool.quote_exact_in(victim, False).amount_out, 100)
    best = best_attack(pool, False, victim, min_out)
    scan = []
    for i in range(1, 400):
        a = best.front_in * i // 200
        r = simulate_attack(pool, False, a, victim)
        if r is not None and r[1] >= min_out:
            scan.append(r[2] - a)
    assert best.gross_profit >= max(scan)


def test_safe_slippage_is_the_exact_boundary():
    pool = v2_pool()
    r = assess_swap(pool, WETH, 10 * E18, slippage_bps=100, gas_price_wei=20 * 10 ** 9)
    s = r.safe_slippage_bps
    assert s is not None and 0 < s < 100
    assert not _profitable(pool, 10 * E18, s, r.gas_cost)
    assert _profitable(pool, 10 * E18, s + 1, r.gas_cost)
    assert any(str(s) in a for a in r.advice)


def test_zero_slippage_leaves_no_room():
    r = assess_swap(v2_pool(), WETH, 10 * E18, slippage_bps=0)
    assert not r.sandwichable and r.max_safe_amount_in is None


def test_small_swap_is_safe_once_gas_counts():
    pool = v2_pool()
    # 6 WETH moves this pool by ~0.6%, just above the two 0.3% fees an attacker pays.
    without_gas = assess_swap(pool, WETH, 6 * E18, slippage_bps=10)
    with_gas = assess_swap(pool, WETH, 6 * E18, slippage_bps=10, gas_price_wei=30 * 10 ** 9)
    assert without_gas.sandwichable and not with_gas.sandwichable
    assert without_gas.safe_slippage_bps == 0  # with free gas any tolerance pays
    assert with_gas.safe_slippage_bps >= 10


def test_victim_smaller_than_twice_the_fee_is_never_sandwichable():
    # 3 WETH moves the pool ~0.3%: less than the attacker's two 0.3% fees.
    r = assess_swap(v2_pool(), WETH, 3 * E18, slippage_bps=5_000)
    assert not r.sandwichable and r.safe_slippage_bps == 5_000


def test_max_safe_amount_is_the_exact_boundary():
    pool = v2_pool()
    gas_price = 20 * 10 ** 9
    r = assess_swap(pool, WETH, 10 * E18, slippage_bps=100, gas_price_wei=gas_price)
    m = r.max_safe_amount_in
    assert m is not None and 0 < m < 10 * E18
    assert not _profitable(pool, m, 100, r.gas_cost)
    assert _profitable(pool, m + 1, 100, r.gas_cost)


def test_non_weth_input_without_price_ignores_gas_conservatively():
    r = assess_swap(v2_pool(), TOKEN, 20_000 * E18, slippage_bps=100, gas_price_wei=20 * 10 ** 9, lang="en")
    assert not r.gas_counted and r.gas_cost == 0
    assert any("Gas was ignored" in n for n in r.notes)
    priced = assess_swap(v2_pool(), TOKEN, 20_000 * E18, slippage_bps=100, gas_price_wei=20 * 10 ** 9,
                         token_in_per_eth=Fraction(2000), lang="en")
    assert priced.gas_counted and priced.gas_cost == 2 * (21_000 + 60_000) * 20 * 10 ** 9 * 2000 // E18


def test_v3_single_range_pool():
    pool = ConcentratedPool(TOKEN, WETH, get_sqrt_ratio_at_tick(-80_000), 10 ** 24, fee_pips=500, tick_spacing=10)
    r = assess_swap(pool, WETH, 20 * E18, slippage_bps=100, lang="en")
    assert r.version == "v3" and r.sandwichable
    assert any("v3 pool without ticks" in n for n in r.notes)
    assert pool.sqrt_price_x96 == get_sqrt_ratio_at_tick(-80_000)


def test_russian_advice_by_default():
    r = assess_swap(v2_pool(), WETH, 10 * E18, slippage_bps=100, lang="ru")
    assert r.advice[0].startswith("Своп можно засэндвичить")


def test_pool_from_spec():
    v2 = pool_from_spec({"version": "v2", "token0": TOKEN, "token1": WETH, "reserve0": "5", "reserve1": 7})
    assert isinstance(v2, V2Pool) and (v2.reserve0, v2.reserve1, v2.fee_bps) == (5, 7, 30)
    v3 = pool_from_spec({"version": "v3", "token0": TOKEN, "token1": WETH, "fee": 3000,
                         "sqrt_price_x96": 2 ** 96, "liquidity": 10 ** 18, "ticks": {"-60": 5, "60": -5}})
    assert v3.tick_spacing == 60 and v3.ticks == {-60: 5, 60: -5}
    with pytest.raises(MASEError):
        pool_from_spec({"version": "v9", "token0": TOKEN, "token1": WETH})


def test_bad_arguments():
    with pytest.raises(MASEError):
        assess_swap(v2_pool(), "0x" + "9" * 40, E18)
    with pytest.raises(MASEError):
        assess_swap(v2_pool(), WETH, 0)
    with pytest.raises(MASEError):
        assess_swap(v2_pool(), WETH, E18, slippage_bps=10_000)


def test_cli_check_swap(tmp_path, capsys):
    spec = {"pool": {"version": "v2", "token0": TOKEN, "token1": WETH, "reserve0": 2_000_000 * E18,
                     "reserve1": 1_000 * E18},
            "token_in": WETH, "amount_in": 10 * E18, "slippage_bps": 100, "gas_price_wei": 20 * 10 ** 9}
    path = tmp_path / "swap.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert cli_main(["--lang", "en", "--check-swap", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["sandwichable"] is True and 0 < out["safe_slippage_bps"] < 100


def test_cli_check_pool_needs_token_and_amount(capsys):
    assert cli_main(["--check-pool", "0x" + "2" * 40]) == 2
