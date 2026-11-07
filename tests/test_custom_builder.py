"""Layer 14: custom bot builder, blueprint schema, bytecode VM, AI synthesis and concurrency."""

from __future__ import annotations

import copy
import json
import pickle
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from simulator import custom_builder as cb
from simulator.__main__ import main
from simulator.ai_agent_factory import AgentFactory
from simulator.attack_vectors import ActionReverted, AttackState
from simulator.config import ConfigError, SimulationConfig
from simulator.controller import run_custom_builder
from simulator.custom_builder import (
    ACTIONS,
    BLUEPRINT_SCHEMA,
    COMPARATORS,
    EVENTS,
    KNOWN_TRIGGER_CLASSES,
    MAX_INSTRUCTIONS,
    METRICS,
    N_BLUEPRINT_GENES,
    OPCODES,
    TEMPLATES,
    BlueprintError,
    BlueprintSynthesizer,
    CustomBot,
    FrozenMap,
    VictimContext,
    assemble,
    blueprint_digest,
    blueprint_from_genes,
    compile_blueprint,
    disassemble,
    evaluate_blueprint,
    normalize_blueprint,
    opcode_trace,
    run_blueprint,
    scenario_for,
    spawn_custom_bot,
    template,
    trigger_class,
    validate_blueprint,
)
from simulator.invariants import input_reserve_scale
from simulator.parallel import WorkerPool
from simulator.slippage_engine import ConcentratedPool, V2Pool, V4Pool

SEED = 5
VICTIMS = 10


@pytest.fixture(scope="module")
def scn():
    return scenario_for(SEED, VICTIMS)


@pytest.fixture(scope="module")
def contexts(scn):
    return CustomBot().contexts(scn)


def bp(script, trigger=None, name="t", **extra):
    out = {"name": name, "script": script}
    if trigger is not None:
        out["trigger"] = trigger
    out.update(extra)
    return out


SANDWICH = [{"op": "frontrun", "size": 5_000}, {"op": "victim"}, {"op": "backrun"}]


def fingerprint(scenario):
    return pickle.dumps((scenario.pools, scenario.markets, scenario.victims))


def manual_sandwich(scenario, ctx, ppm):
    st = AttackState(scenario)
    pool = st.pools[ctx.pool.pool_id]
    amount = input_reserve_scale(pool, ctx.zero_for_one) * ppm // 1_000_000
    st.swap(pool.pool_id, ctx.tx.token_in, amount, label="frontrun")
    st.victim(ctx.tx)
    st.swap(pool.pool_id, ctx.token_out, st.ledger[ctx.token_out], label="backrun")
    st.settle()
    return st.result("manual", "")


# ---------------------------------------------------------------------------
# Schema: templates, defaults, digests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_template_normalizes_and_compiles(name):
    norm = normalize_blueprint(template(name))
    assert norm["schema"] == 1 and norm["name"] == name
    cls = compile_blueprint(norm)
    assert issubclass(cls, CustomBot) and cls.NAME == name


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_normalization_is_idempotent(name):
    once = normalize_blueprint(template(name))
    assert normalize_blueprint(copy.deepcopy(once)) == once


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_template_trigger_class_is_known(name):
    assert trigger_class(normalize_blueprint(template(name))["trigger"]) in KNOWN_TRIGGER_CLASSES


def test_template_returns_independent_copies():
    a = template("classic_sandwich")
    a["script"].clear()
    assert TEMPLATES["classic_sandwich"]["script"]


def test_unknown_template_raises():
    with pytest.raises(BlueprintError):
        template("nope")


def test_defaults_are_filled_in():
    norm = normalize_blueprint(bp([{"op": "frontrun"}, {"op": "victim"}, {"op": "backrun"}]))
    assert norm["trigger"] == {"const": True}
    assert norm["tracing"] == {"opcodes": False, "sequential": False, "record": False}
    assert norm["script"][0] == {"op": "frontrun", "size": {"ppm": 5_000}}
    assert norm["pipeline"] == cb.PIPELINE_DEFAULTS


def test_integer_size_becomes_ppm_spec():
    norm = normalize_blueprint(bp([{"op": "swap", "pool": 1, "side": 0, "size": 77}]))
    assert norm["script"][0]["size"] == {"ppm": 77}


def test_digest_is_stable_and_name_sensitive():
    a = template("classic_sandwich")
    assert blueprint_digest(a) == blueprint_digest(copy.deepcopy(a))
    b = dict(a, name="other")
    assert blueprint_digest(a) != blueprint_digest(b)


def test_digest_ignores_spelled_out_defaults():
    a = bp(SANDWICH)
    b = bp(SANDWICH, pipeline=dict(cb.PIPELINE_DEFAULTS), tracing={"opcodes": False})
    assert blueprint_digest(a) == blueprint_digest(b)


def test_schema_document_lists_every_vocabulary():
    assert set(BLUEPRINT_SCHEMA["metrics"]) == set(METRICS)
    assert BLUEPRINT_SCHEMA["actions"] == list(ACTIONS)
    assert BLUEPRINT_SCHEMA["opcodes"] == list(OPCODES)
    assert set(BLUEPRINT_SCHEMA["events"]) == set(EVENTS)
    json.dumps(BLUEPRINT_SCHEMA)


def test_validate_returns_empty_for_valid():
    assert validate_blueprint(template("composite_bridge_bundle")) == []


INVALID = [
    ("not a dict", [], "$"),
    ("missing name", {"script": SANDWICH}, "$.name"),
    ("bad name", bp(SANDWICH, name="bad name!"), "$.name"),
    ("unknown key", dict(bp(SANDWICH), extra=1), "$"),
    ("schema 2", dict(bp(SANDWICH), schema=2), "$.schema"),
    ("empty script", bp([]), "$.script"),
    ("script not list", bp({"op": "victim"}), "$.script"),
    ("unknown op", bp([{"op": "teleport"}]), "$.script[0].op"),
    ("action no op", bp([{"size": 1}]), "$.script[0]"),
    ("size zero", bp([{"op": "frontrun", "size": 0}]), "$.script[0].size.ppm"),
    ("size too big", bp([{"op": "frontrun", "size": {"ppm": 2_000_000}}]), "$.script[0].size.ppm"),
    ("bad spec", bp([{"op": "frontrun", "size": {"weird": 1}}]), "$.script[0].size"),
    ("bool size", bp([{"op": "frontrun", "size": True}]), "$.script[0].size"),
    ("optimize reversed", bp([{"op": "frontrun", "size": {"optimize": [10, 5]}}]), "$.script[0].size.optimize"),
    ("two optimizers", bp([{"op": "frontrun", "size": {"optimize": [1, 5]}}, {"op": "victim"},
                           {"op": "swap", "size": {"optimize": [1, 5]}}]), "$.script[2].size"),
    ("victims max", bp([{"op": "victims", "max": 0}]), "$.script[0].max"),
    ("bad side", bp([{"op": "swap", "side": 2}]), "$.script[0].side"),
    ("bad pool", bp([{"op": "swap", "pool": -1}]), "$.script[0].pool"),
    ("mint pct", bp([{"op": "mint", "pct": 0}]), "$.script[0].pct"),
    ("arb hops", bp([{"op": "arb", "max_hops": 4}]), "$.script[0].max_hops"),
    ("arb tload", bp([{"op": "arb", "size": {"tload": "x"}}]), "$.script[0].size"),
    ("tstore key", bp([{"op": "tstore", "key": "1bad"}]), "$.script[0].key"),
    ("tswap same", bp([{"op": "tswap", "a": "x", "b": "x"}]), "$.script[0]"),
    ("repeat zero", bp([{"op": "repeat", "times": 0, "body": [{"op": "victim"}]}]), "$.script[0].times"),
    ("repeat empty", bp([{"op": "repeat", "times": 2, "body": []}]), "$.script[0].body"),
    ("atomic mode", bp([{"op": "atomic", "on_revert": "maybe", "body": [{"op": "victim"}]}]),
     "$.script[0].on_revert"),
    ("bridge victim", bp([{"op": "bridge", "body": [{"op": "victim"}]}]), "$.script[0].body[0].op"),
    ("bridge nested", bp([{"op": "bridge", "body": [{"op": "bridge", "body": [{"op": "arb", "pool": 0}]}]}]),
     "$.script[0].body[0].op"),
    ("bridge victim pool", bp([{"op": "bridge", "body": [{"op": "arb", "pool": "$victim"}]}]),
     "$.script[0].body[0].pool"),
    ("bridge fee", bp([{"op": "bridge", "fee_bps": 5_000, "body": [{"op": "arb", "pool": 0}]}]),
     "$.script[0].fee_bps"),
    ("unknown metric", bp(SANDWICH, {"metric": "moon_phase", "value": 1}), "$.trigger.metric"),
    ("bad comparator", bp(SANDWICH, {"metric": "ticks_crossed", "op": "~", "value": 1}), "$.trigger.op"),
    ("between shape", bp(SANDWICH, {"metric": "ticks_crossed", "op": "between", "value": [3, 1]}),
     "$.trigger.value"),
    ("in shape", bp(SANDWICH, {"metric": "ticks_crossed", "op": "in", "value": []}), "$.trigger.value"),
    ("opcode without tracing", bp(SANDWICH, {"opcode": "CALL", "value": 1}), "$.tracing.opcodes"),
    ("unknown opcode", bp(SANDWICH, {"opcode": "JUMPDEST", "value": 1}, tracing={"opcodes": True}),
     "$.trigger.opcode"),
    ("unknown event", bp(SANDWICH, {"event": "Rugpull"}), "$.trigger.event"),
    ("event param", bp(SANDWICH, {"event": "LargeSwap", "min_ppm": 0}), "$.trigger.min_ppm"),
    ("tload in trigger", bp(SANDWICH, {"tload": "x", "value": 1}), "$.trigger"),
    ("empty all", bp(SANDWICH, {"all": []}), "$.trigger.all"),
    ("const not bool", bp(SANDWICH, {"const": 1}), "$.trigger.const"),
    ("pipeline type", bp(SANDWICH, pipeline={"settle": "yes"}), "$.pipeline.settle"),
    ("pipeline range", bp(SANDWICH, pipeline={"max_attempts": 0}), "$.pipeline.max_attempts"),
    ("pipeline revert", bp(SANDWICH, pipeline={"on_revert": "pray"}), "$.pipeline.on_revert"),
    ("pipeline key", bp(SANDWICH, pipeline={"turbo": True}), "$.pipeline"),
    ("tracing type", bp(SANDWICH, tracing={"record": 1}), "$.tracing.record"),
]


@pytest.mark.parametrize("label,raw,path", INVALID, ids=[c[0] for c in INVALID])
def test_invalid_blueprint_is_rejected_with_its_path(label, raw, path):
    with pytest.raises(BlueprintError) as info:
        normalize_blueprint(raw)
    assert any(p == path or p.startswith(path) for p, _ in info.value.issues), info.value.issues
    assert isinstance(info.value, ConfigError)


def test_errors_collect_every_issue():
    raw = {"name": "x y", "script": [{"op": "nope"}, {"op": "frontrun", "size": 0}]}
    issues = validate_blueprint(raw)
    assert {p for p, _ in issues} >= {"$.name", "$.script[0].op", "$.script[1].size.ppm"}


def test_depth_limit():
    node = {"const": True}
    for _ in range(cb.MAX_DEPTH + 1):
        node = {"not": node}
    assert any("deeper" in m for _, m in validate_blueprint(bp(SANDWICH, node)))


def test_trigger_node_limit():
    nodes = [{"const": True}] * cb.MAX_CHILDREN
    tree = {"all": [{"any": nodes} for _ in range(5)]}
    assert any("trigger nodes" in m for _, m in validate_blueprint(bp(SANDWICH, tree)))


def test_instruction_limit_blocks_repeat_explosion():
    body = [{"op": "tstore", "key": "k", "value": 1}]
    for _ in range(4):
        body = [{"op": "repeat", "times": 8, "body": body}]
    issues = validate_blueprint(bp(body))
    assert any("instructions" in m for _, m in issues)


def test_action_limit():
    issues = validate_blueprint(bp([{"op": "victim"}] * (cb.MAX_SCRIPT_ACTIONS + 1)))
    assert any("actions" in m for _, m in issues)


# ---------------------------------------------------------------------------
# Trigger compiler and VM
# ---------------------------------------------------------------------------


CMP_CASES = [(5, "<", 6, True), (6, "<", 6, False), (6, "<=", 6, True), (7, ">", 6, True), (6, ">", 6, False),
             (6, ">=", 6, True), (6, "==", 6, True), (5, "!=", 6, True), (5, "between", (5, 9), True),
             (10, "between", (5, 9), False), (3, "in", (1, 3), True), (2, "in", (1, 3), False)]


@pytest.mark.parametrize("lhs,op,value,expected", CMP_CASES)
def test_comparators(lhs, op, value, expected):
    assert cb._compare(lhs, op, value) is expected


def test_comparator_catalog():
    assert set(COMPARATORS) == {c[1] for c in CMP_CASES}


def test_trigger_class_is_order_independent():
    a = {"all": [{"metric": "ticks_crossed", "op": ">", "value": 1}, {"event": "Swap"}]}
    b = {"all": [{"event": "Swap"}, {"metric": "ticks_crossed", "op": ">", "value": 9}]}
    assert trigger_class(a) == trigger_class(b) == "all(event:Swap,metric:ticks_crossed>)"


def test_trigger_class_distinguishes_structure():
    a = {"any": [{"event": "Swap"}, {"event": "TickCross"}]}
    b = {"all": [{"event": "Swap"}, {"event": "TickCross"}]}
    assert trigger_class(a) != trigger_class(b)
    assert trigger_class({"not": {"event": "Swap"}}) == "not(event:Swap)"


def test_trigger_assembles_to_postfix():
    prog = assemble(bp(SANDWICH, {"all": [{"const": True}, {"not": {"event": "Swap"}}]}))
    kinds = [i[0] for i in prog.trigger_code]
    assert kinds == ["CONST", "EVENT", "NOT", "AND"]
    assert prog.trigger_code[-1] == ("AND", 2)


def ref_eval(node, ctx, sequential=False):
    if "all" in node:
        return all(ref_eval(k, ctx, sequential) for k in node["all"])
    if "any" in node:
        return any(ref_eval(k, ctx, sequential) for k in node["any"])
    if "not" in node:
        return not ref_eval(node["not"], ctx, sequential)
    if "const" in node:
        return node["const"]
    if "metric" in node:
        return cb._compare(ctx.metric(node["metric"]), node["op"], cb._freeze(node["value"]))
    if "opcode" in node:
        return cb._compare(ctx.trace().count(node["opcode"]), node["op"], cb._freeze(node["value"]))
    if "opcode_seq" in node:
        return cb._subsequence(node["opcode_seq"], ctx.trace(), sequential)
    params = {k: v for k, v in node.items() if k != "event"}
    return ctx.event(node["event"], cb._freeze(params))


def random_tree(rng, ctxs, depth=0):
    roll = rng.random()
    if depth < 3 and roll < 0.3:
        kind = rng.choice(("all", "any"))
        return {kind: [random_tree(rng, ctxs, depth + 1) for _ in range(rng.randint(1, 3))]}
    if depth < 3 and roll < 0.4:
        return {"not": random_tree(rng, ctxs, depth + 1)}
    roll = rng.random()
    if roll < 0.5:
        name = rng.choice(sorted(METRICS))
        sample = ctxs[rng.randrange(len(ctxs))].metric(name)
        op = rng.choice(("<", "<=", ">", ">=", "==", "!="))
        return {"metric": name, "op": op, "value": sample}
    if roll < 0.7:
        name = rng.choice(sorted(EVENTS))
        return {"event": name}
    if roll < 0.85:
        return {"opcode": rng.choice(OPCODES), "op": rng.choice((">=", "<", "==")), "value": rng.randint(0, 4)}
    if roll < 0.95:
        return {"opcode_seq": [rng.choice(OPCODES) for _ in range(rng.randint(1, 3))]}
    return {"const": rng.random() < 0.5}


@pytest.mark.parametrize("seed", range(24))
def test_vm_trigger_matches_reference_interpreter(seed, scn, contexts):
    rng = random.Random(f"tree:{seed}")
    tree = random_tree(rng, contexts)
    sequential = bool(seed % 2)
    bot = spawn_custom_bot(bp(SANDWICH, tree, name=f"tree{seed}",
                              tracing={"opcodes": True, "sequential": sequential}))
    for ctx in contexts:
        assert bot.fires(ctx) == ref_eval(normalize_blueprint(bot.blueprint())["trigger"], ctx, sequential)


@pytest.mark.parametrize("name", sorted(METRICS))
def test_every_metric_is_a_deterministic_integer(name, scn, contexts):
    values = [c.metric(name) for c in contexts]
    again = [VictimContext(scn, c.tx, c.position).metric(name) for c in contexts]
    assert values == again and all(isinstance(v, int) for v in values)


def test_metric_definitions(scn, contexts):
    for ctx in contexts:
        assert ctx.metric("victim_amount_in") == ctx.tx.amount_in
        assert ctx.metric("pool_version") in (2, 3, 4)
        assert ctx.metric("pool_side") in (0, 1)
        assert 0 <= ctx.metric("victim_rank") < ctx.metric("same_side_victims")
        assert ctx.metric("victim_slippage_bps") >= 0
        assert ctx.metric("gas_price_wei") == scn.gas_price_wei
        assert scn.pools[ctx.metric("pool_index")].pool_id == ctx.tx.route[0]


def test_unknown_metric_raises(contexts):
    with pytest.raises(BlueprintError):
        contexts[0].metric("nope")


@pytest.mark.parametrize("event", sorted(EVENTS))
def test_events_follow_their_metrics(event, contexts):
    params = cb._freeze({p: d for p, (d, _, _) in EVENTS[event].items()})
    for ctx in contexts:
        got = ctx.event(event, params)
        if event == "Swap":
            assert got is True
        elif event == "LargeSwap":
            assert got == (ctx.metric("victim_size_ppm") >= 1_000)
        elif event == "PriceDeviation":
            assert got == (ctx.metric("cross_venue_spread_bps") >= 10)
        elif event == "LoanAtRisk":
            assert got == (ctx.metric("loan_min_health_bps") < 10_100)
        else:
            assert got == (ctx.metric("ticks_crossed") >= 1)


def test_swap_event_filters_pool_and_side(contexts):
    ctx = contexts[0]
    assert ctx.event("Swap", cb._freeze({"pool": ctx.pool_index, "side": ctx.side}))
    assert not ctx.event("Swap", cb._freeze({"pool": ctx.pool_index, "side": 1 - ctx.side}))


def test_opcode_trace_model_per_version(scn):
    by_type = {type(p): p for p in scn.pools}
    v2 = opcode_trace(by_type[V2Pool])
    assert v2 == ("CALL", "SLOAD", "SLOAD", "CALL", "SSTORE", "SSTORE", "LOG1", "LOG3")
    v3 = opcode_trace(by_type[ConcentratedPool], 2)
    assert v3.count("SSTORE") == 3 and "TSTORE" not in v3
    v4 = opcode_trace(by_type[V4Pool], 0)
    assert v4.count("TSTORE") == 3 and v4[0] == "CALL"


def test_opcode_trace_ticks_are_capped(scn):
    pool = next(p for p in scn.pools if type(p) is ConcentratedPool)
    assert opcode_trace(pool, 10_000) == opcode_trace(pool, 64)


@pytest.mark.parametrize("needle,contiguous,expected", [
    (["CALL", "SSTORE"], False, True), (["SLOAD", "SSTORE"], True, False), (["SSTORE", "SSTORE"], True, True),
    (["LOG3", "CALL"], False, False), (["SLOAD", "CALL", "LOG1"], False, True)])
def test_opcode_subsequence(needle, contiguous, expected):
    hay = ("CALL", "SLOAD", "SLOAD", "CALL", "SSTORE", "SSTORE", "LOG1", "LOG3")
    assert cb._subsequence(needle, hay, contiguous) is expected


def test_multi_hop_victims_are_not_candidates(scn):
    tx = scn.victims[0]
    multi = type(tx)(tx.tx_hash, tx.sender, dict(tx.payload, route=[scn.pools[0].pool_id, scn.pools[1].pool_id]),
                     tx.gas_limit, tx.priority_fee_wei)
    with pytest.raises(ActionReverted):
        VictimContext(scn, multi)


# ---------------------------------------------------------------------------
# Script assembly
# ---------------------------------------------------------------------------


def test_if_else_assembles_forward_jumps():
    prog = assemble(bp([{"op": "if", "cond": {"balance": "$base", "value": 1},
                         "then": [{"op": "victim"}], "else": [{"op": "burn"}, {"op": "burn"}]}]))
    code = prog.script_code
    assert code[0][0] == "JUMPF" and code[0][2] == 3
    assert code[2] == ("JUMP", 5) and code[-1] == ("HALT",)


def test_repeat_is_unrolled():
    prog = assemble(bp([{"op": "repeat", "times": 3, "body": [{"op": "victim"}, {"op": "burn"}]}]))
    assert [i[1] for i in prog.script_code[:-1]] == ["victim", "burn"] * 3


def test_atomic_is_balanced():
    prog = assemble(bp([{"op": "atomic", "body": [{"op": "atomic", "on_revert": "abort", "body": [{"op": "burn"}]}]}]))
    kinds = [i[0] for i in prog.script_code]
    assert kinds == ["ATOMIC", "ATOMIC", "ACT", "COMMIT", "COMMIT", "HALT"]
    assert prog.script_code[0] == ("ATOMIC", 5, "rollback") and prog.script_code[1] == ("ATOMIC", 4, "abort")


def test_bridge_has_its_own_program():
    prog = assemble(template("composite_bridge_bundle"))
    bridge = next(i for i in prog.script_code if i[0] == "BRIDGE")
    assert bridge[2][-1] == ("HALT",) and prog.model


def test_program_flags():
    assert assemble(template("classic_sandwich")).optimize == (50, 200_000)
    assert not assemble(template("classic_sandwich")).model
    assert assemble(bp([{"op": "liquidate"}])).model
    assert assemble(bp([{"op": "arb", "pool": 1}])).victim_free
    assert not assemble(bp(SANDWICH)).victim_free


def test_verify_program_rejects_backward_jump():
    with pytest.raises(BlueprintError):
        cb._verify_program([("JUMP", 0), ("HALT",)])
    with pytest.raises(BlueprintError):
        cb._verify_program([("COMMIT",), ("HALT",)])
    with pytest.raises(BlueprintError):
        cb._verify_program([("ATOMIC", 2, "rollback"), ("HALT",)])


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_disassembly_covers_every_instruction(name):
    cls = compile_blueprint(template(name))
    listing = cls.disassemble()
    assert len(listing["trigger"]) == len(cls.TRIGGER_CODE)
    assert len(listing["script"]) >= len(cls.SCRIPT_CODE)
    assert listing["script"][-1].endswith("HALT")


def test_code_bytes_are_deterministic():
    a = assemble(template("composite_bridge_bundle")).code_bytes()
    assert a == assemble(template("composite_bridge_bundle")).code_bytes()
    assert json.loads(a)


def test_disassemble_free_function():
    lines = disassemble((("CONST", True), ("NOT",)))
    assert lines == ["0000 CONST   true", "0001 NOT"]


# ---------------------------------------------------------------------------
# Execution semantics
# ---------------------------------------------------------------------------


def first_sandwichable(scn, contexts, ppm=5_000):
    for ctx in contexts:
        try:
            return ctx, manual_sandwich(scn, ctx, ppm)
        except ActionReverted:
            continue
    pytest.skip("no sandwichable victim")


def test_compiled_sandwich_equals_manual_to_the_wei(scn, contexts):
    ctx, manual = first_sandwichable(scn, contexts)
    out = spawn_custom_bot(bp(SANDWICH, pipeline={"min_profit_base": 0, "require_flat": False})).execute(scn, ctx)
    assert out.result is not None
    assert out.result.net_profit_base == manual.net_profit_base
    assert out.result.ledger == manual.ledger and out.result.gas_used == manual.gas_used


def test_transient_backrun_equals_plain_backrun(scn, contexts):
    ctx, manual = first_sandwichable(scn, contexts)
    script = [{"op": "frontrun", "size": 5_000}, {"op": "tstore", "key": "o", "value": {"last_out": True}},
              {"op": "victim"}, {"op": "swap", "pool": "$victim", "side": "$counter", "size": {"tload": "o"}}]
    out = spawn_custom_bot(bp(script, pipeline={"min_profit_base": 0, "require_flat": False})).execute(scn, ctx)
    assert out.result.ledger == manual.ledger
    # transient storage costs a little gas on top of the identical token flow
    assert out.result.gas_used == manual.gas_used + 2 * cb.GAS_TRANSIENT


def test_optimized_plan_beats_or_matches_fixed(scn, contexts):
    ctx, _ = first_sandwichable(scn, contexts)
    fixed = spawn_custom_bot(bp(SANDWICH, pipeline={"min_profit_base": 0})).plan(scn, ctx)
    opt = spawn_custom_bot(template("classic_sandwich")).plan(scn, ctx)
    if fixed.valid:
        assert opt.valid and opt.net_profit_base >= fixed.net_profit_base or opt.param is not None


def test_atomic_rollback_restores_state(scn, contexts):
    ctx, _ = first_sandwichable(scn, contexts)
    pipe = {"min_profit_base": 0, "require_flat": False}
    plain = spawn_custom_bot(bp(SANDWICH, pipeline=pipe, name="plain")).execute(scn, ctx)
    guarded = SANDWICH + [{"op": "atomic", "body": [{"op": "swap", "size": 1_000},
                                                    {"op": "tstore", "key": "x", "value": 7}, {"op": "burn"}]}]
    out = spawn_custom_bot(bp(guarded, pipeline=pipe, name="guarded")).execute(scn, ctx)
    assert out.result.ledger == plain.result.ledger and out.result.gas_used == plain.result.gas_used
    assert out.result.steps == plain.result.steps


def test_atomic_abort_fails_the_plan(scn, contexts):
    ctx, _ = first_sandwichable(scn, contexts)
    script = SANDWICH + [{"op": "atomic", "on_revert": "abort", "body": [{"op": "burn"}]}]
    out = spawn_custom_bot(bp(script)).execute(scn, ctx)
    assert not out.valid and "position" in out.error


def test_inner_abort_propagates_to_outer_rollback(scn, contexts):
    ctx, _ = first_sandwichable(scn, contexts)
    pipe = {"min_profit_base": 0, "require_flat": False}
    plain = spawn_custom_bot(bp(SANDWICH, pipeline=pipe, name="p2")).execute(scn, ctx)
    script = SANDWICH + [{"op": "atomic", "body": [{"op": "swap", "size": 500},
                                                   {"op": "atomic", "on_revert": "abort", "body": [{"op": "burn"}]}]}]
    out = spawn_custom_bot(bp(script, pipeline=pipe, name="nested")).execute(scn, ctx)
    assert out.result.ledger == plain.result.ledger


def test_on_revert_skip_continues(scn, contexts):
    ctx, manual = first_sandwichable(scn, contexts)
    script = [{"op": "burn"}] + SANDWICH
    pipe = {"min_profit_base": 0, "require_flat": False}
    aborted = spawn_custom_bot(bp(script, pipeline=pipe, name="ab")).execute(scn, ctx)
    skipped = spawn_custom_bot(bp(script, pipeline=dict(pipe, on_revert="skip"), name="sk")).execute(scn, ctx)
    assert not aborted.valid
    assert skipped.result.ledger == manual.ledger


def test_pipeline_override_skip_without_recompiling(scn, contexts):
    ctx, manual = first_sandwichable(scn, contexts)
    cls = compile_blueprint(bp([{"op": "burn"}] + SANDWICH, name="ovr", pipeline={"require_flat": False,
                                                                                  "min_profit_base": 0}))
    assert not cls().execute(scn, ctx).valid
    assert cls({"on_revert": "skip"}).execute(scn, ctx).result.ledger == manual.ledger


def test_tswap_and_if_branching(scn, contexts):
    ctx, _ = first_sandwichable(scn, contexts)
    script = [{"op": "tstore", "key": "a", "value": 5}, {"op": "tswap", "a": "a", "b": "b"},
              {"op": "if", "cond": {"tload": "b", "op": "==", "value": 5},
               "then": SANDWICH, "else": [{"op": "burn"}]}]
    out = spawn_custom_bot(bp(script, pipeline={"min_profit_base": 0, "require_flat": False})).execute(scn, ctx)
    assert out.result is not None and out.result.victims == [ctx.tx.tx_hash]


def test_if_else_branch(scn, contexts):
    ctx, _ = first_sandwichable(scn, contexts)
    script = [{"op": "if", "cond": {"tload": "never", "op": ">", "value": 0}, "then": [{"op": "victim"}],
               "else": [{"op": "burn"}]}]
    out = spawn_custom_bot(bp(script)).execute(scn, ctx)
    assert not out.valid and "position" in out.error


def test_transient_storage_starts_empty_every_run(scn, contexts):
    ctx, _ = first_sandwichable(scn, contexts)
    script = [{"op": "if", "cond": {"tload": "seen", "op": "==", "value": 0}, "then": SANDWICH,
               "else": [{"op": "burn"}]}, {"op": "tstore", "key": "seen", "value": 1}]
    bot = spawn_custom_bot(bp(script, pipeline={"min_profit_base": 0, "require_flat": False}))
    first, second = bot.execute(scn, ctx), bot.execute(scn, ctx)
    assert first.result.ledger == second.result.ledger


def test_repeat_equals_written_out(scn, contexts):
    ctx, _ = first_sandwichable(scn, contexts)
    pipe = {"min_profit_base": 0, "require_flat": False}
    rep = [{"op": "repeat", "times": 2, "body": [{"op": "frontrun", "size": 1_000}]}, {"op": "victim"},
           {"op": "backrun"}]
    flat = [{"op": "frontrun", "size": 1_000}, {"op": "frontrun", "size": 1_000}, {"op": "victim"},
            {"op": "backrun"}]
    a = spawn_custom_bot(bp(rep, pipeline=pipe, name="r")).execute(scn, ctx)
    b = spawn_custom_bot(bp(flat, pipeline=pipe, name="f")).execute(scn, ctx)
    assert (a.valid, a.net_profit_base) == (b.valid, b.net_profit_base)


def test_victims_action_adds_same_side_victims(scn, contexts):
    ctx = max(contexts, key=lambda c: c.metric("same_side_victims"))
    if ctx.metric("same_side_victims") < 2:
        pytest.skip("no pool with two same-side victims")
    script = [{"op": "victim"}, {"op": "victims", "max": 8}]
    out = spawn_custom_bot(bp(script, pipeline={"settle": False, "require_flat": False,
                                                "min_profit_base": 0})).execute(scn, ctx)
    ref = AttackState(scn)
    ref.victim(ctx.tx)
    for tx in ctx._same_side():
        if tx.tx_hash != ctx.tx.tx_hash and len(ref.victims_executed) < 9:
            try:
                ref.victim(tx)
            except ActionReverted:
                continue
    assert out.result is not None and out.result.victims == ref.victims_executed
    assert out.result.victims[0] == ctx.tx.tx_hash and len(set(out.result.victims)) == len(out.result.victims)


def test_victim_twice_reverts(scn, contexts):
    out = spawn_custom_bot(bp([{"op": "victim"}, {"op": "victim"}])).execute(scn, contexts[0])
    assert not out.valid and "already" in out.error


def test_victim_actions_need_a_victim(scn):
    out = spawn_custom_bot(bp([{"op": "victim"}])).execute(scn, None)
    assert not out.valid and "victim" in out.error


def test_arb_without_victim(scn):
    out = spawn_custom_bot(bp([{"op": "arb", "pool": 0, "size": 500}], pipeline={"min_profit_base": 0})).execute(
        scn, None)
    assert out.valid or out.error


def test_pool_index_out_of_range_reverts(scn):
    out = spawn_custom_bot(bp([{"op": "swap", "pool": 60, "side": 0, "size": 10}])).execute(scn, None)
    assert not out.valid and "pool" in out.error


def test_unknown_token_reverts(scn):
    script = [{"op": "tstore", "key": "k", "value": {"balance": "XYZ"}}]
    out = spawn_custom_bot(bp(script)).execute(scn, None)
    assert not out.valid and "token" in out.error


def test_gas_cap_override(scn, contexts):
    ctx, _ = first_sandwichable(scn, contexts)
    out = spawn_custom_bot(bp(SANDWICH), {"max_gas": 21_000}).execute(scn, ctx)
    assert not out.valid and out.error == "gas cap"


def test_min_profit_override(scn, contexts):
    run = spawn_custom_bot(template("classic_sandwich"), {"min_profit_base": 10**30}).run(scn)
    assert run.valid == [] and run.net_profit_base == 0


def test_invalid_override_is_rejected():
    with pytest.raises(BlueprintError):
        spawn_custom_bot(template("classic_sandwich"), {"max_attempts": 0})
    with pytest.raises(BlueprintError):
        spawn_custom_bot(template("classic_sandwich"), {"warp": 1})


@pytest.mark.parametrize("attempts", [1, 2, 4])
def test_max_attempts_bounds_planning(scn, attempts):
    run = spawn_custom_bot(template("multi_victim_sandwich"), {"max_attempts": attempts}).run(scn)
    assert run.attempted == min(attempts, run.fired) and len(run.plans) == run.attempted


def test_empty_plan_is_invalid(scn, contexts):
    out = spawn_custom_bot(bp([{"op": "tstore", "key": "k", "value": 1}])).execute(scn, contexts[0])
    assert not out.valid and out.error == "empty plan"


def test_bridge_leg_is_a_model_with_fees(scn):
    run = run_blueprint(template("composite_bridge_bundle"), seed=1, victims=VICTIMS)
    legs = [leg for p in run.valid for leg in p.result.extra.get("bridges", [])]
    assert legs, "expected at least one bridged plan on seed 1"
    for p in run.valid:
        if p.result.extra.get("bridges"):
            assert p.result.status == "model"
    for leg in legs:
        assert leg["fee"] == -(-leg["amount"] * 5 // 10_000)
        assert leg["delay_blocks"] == 2 and leg["remote_steps"] >= 1


def test_bridge_without_balance_reverts(scn, contexts):
    script = [{"op": "bridge", "token": "DAI", "body": [{"op": "arb", "pool": 0}]}]
    out = spawn_custom_bot(bp(script)).execute(scn, contexts[0])
    assert not out.valid and "balance" in out.error


def test_remote_scenario_is_deterministic():
    assert cb._remote_seed(3, "l2") == cb._remote_seed(3, "l2") != cb._remote_seed(3, "l3")


def test_recorded_opcode_trace(scn):
    run = run_blueprint(template("composite_bridge_bundle"), seed=SEED, victims=VICTIMS)
    best = run.best
    if best is None:
        pytest.skip("no valid plan")
    trace = best.result.extra["opcode_trace"]
    assert len(trace) == len(best.result.steps) and all(isinstance(t, list) and t for t in trace)


def test_liquidation_is_marked_model(scn):
    market = scn.markets[0]
    pi = next(i for i, p in enumerate(scn.pools) if p.pool_id == market.oracle_pool_id)
    side = scn.pools[pi].tokens().index(market.collateral_token)
    script = [{"op": "swap", "pool": pi, "side": side, "size": 60_000}, {"op": "liquidate", "repay_ppm": 500_000}]
    out = spawn_custom_bot(bp(script, pipeline={"min_profit_base": 0, "require_flat": False,
                                                "settle": False})).execute(scn, None)
    assert out.result is not None, out.error
    assert out.result.status == "model" and any(s["op"] == "liquidate" for s in out.result.steps)


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_template_runs_are_deterministic(name):
    a = run_blueprint(template(name), seed=2, victims=VICTIMS).summary()
    b = run_blueprint(template(name), seed=2, victims=VICTIMS).summary()
    assert a == b


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_valid_plans_satisfy_the_pipeline(name, scn):
    bot = spawn_custom_bot(template(name))
    run = bot.run(scn)
    for p in run.valid:
        r = p.result
        assert r.flat and r.net_profit_base >= bot.pipeline["min_profit_base"]
        assert r.gas_used <= bot.pipeline["max_gas"] and r.capital_used_base <= scn.capital_base
        assert r.extra["digest"] == bot.DIGEST


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_runs_never_mutate_the_scenario(name, scn):
    before = fingerprint(scn)
    spawn_custom_bot(template(name)).run(scn)
    assert fingerprint(scn) == before


def test_evaluate_blueprint_sums_seeds():
    row = evaluate_blueprint(template("classic_sandwich"), (1, 2), VICTIMS)
    singles = [run_blueprint(template("classic_sandwich"), seed=s, victims=VICTIMS).net_profit_base for s in (1, 2)]
    assert row["valid"] and row["net_profit_base"] == sum(singles) and len(row["runs"]) == 2


def test_evaluate_invalid_blueprint_scores_zero():
    row = evaluate_blueprint({"name": "x"}, (1,), VICTIMS)
    assert not row["valid"] and row["net_profit_base"] == 0 and row["error"]


def test_plan_outcome_and_run_serialize(scn):
    run = spawn_custom_bot(template("classic_sandwich")).run(scn)
    json.dumps(run.summary())
    for p in run.plans:
        json.dumps(p.to_dict(), default=str)


# ---------------------------------------------------------------------------
# Compiled classes and the class cache
# ---------------------------------------------------------------------------


def test_same_blueprint_same_class():
    assert compile_blueprint(template("jit_liquidity")) is compile_blueprint(template("jit_liquidity"))


def test_different_blueprint_different_class():
    a = compile_blueprint(template("jit_liquidity"))
    b = compile_blueprint(dict(template("jit_liquidity"), name="jit2"))
    assert a is not b and a.DIGEST != b.DIGEST


def test_class_name_and_module():
    cls = compile_blueprint(template("classic_sandwich"))
    assert cls.__name__ == f"Custom_classic_sandwich_{cls.DIGEST[:12]}"
    assert cls.__module__ == "simulator.custom_builder"


def test_class_attributes_are_immutable():
    cls = compile_blueprint(template("composite_bridge_bundle"))

    def walk(obj):
        assert not isinstance(obj, (list, dict, set)), type(obj)
        if isinstance(obj, tuple):
            for v in obj:
                walk(v)

    walk(cls.TRIGGER_CODE)
    walk(cls.SCRIPT_CODE)
    walk(cls.PIPELINE)
    assert isinstance(cls.BLUEPRINT, str)


def test_frozen_map():
    fm = cb._freeze({"b": [1, {"c": 2}], "a": 1})
    assert isinstance(fm, FrozenMap) and fm.keys() == ["a", "b"]
    assert fm.get("b")[1].get("c") == 2 and fm.get("zz", 9) == 9
    assert fm.to_dict() == {"a": 1, "b": [1, {"c": 2}]}
    assert hash(fm) == hash(cb._freeze({"a": 1, "b": [1, {"c": 2}]}))
    assert pickle.loads(pickle.dumps(fm)) == fm


def test_compiled_program_pickles():
    prog = assemble(template("composite_bridge_bundle"))
    assert pickle.loads(pickle.dumps(prog)) == prog


def test_blueprint_roundtrip_from_class():
    cls = compile_blueprint(template("transient_sandwich"))
    assert compile_blueprint(cls.blueprint()) is cls


def test_cache_info_and_clear():
    cb.clear_class_cache()
    compile_blueprint(template("classic_sandwich"))
    compile_blueprint(template("classic_sandwich"))
    info = cb.cache_info()
    assert info == {"classes": 1, "hits": 1, "misses": 1}
    cb.clear_class_cache()
    assert cb.cache_info()["classes"] == 0


# ---------------------------------------------------------------------------
# Concurrency and memory isolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("threads", [2, 4, 8, 16, 32])
def test_concurrent_compilation_yields_one_class(threads):
    raw = dict(template("composite_bridge_bundle"), name=f"race{threads}")
    barrier = threading.Barrier(threads)

    def work(_):
        barrier.wait()
        return compile_blueprint(copy.deepcopy(raw))

    with ThreadPoolExecutor(threads) as ex:
        classes = list(ex.map(work, range(threads)))
    assert all(c is classes[0] for c in classes)


@pytest.mark.parametrize("threads", [2, 4, 8, 16])
def test_concurrent_runs_match_serial(threads, scn):
    names = sorted(TEMPLATES)
    serial = {n: spawn_custom_bot(template(n)).run(scn).summary() for n in names}
    jobs = [names[i % len(names)] for i in range(threads * 2)]
    barrier = threading.Barrier(threads)

    def work(name):
        try:
            barrier.wait(timeout=5)
        except threading.BrokenBarrierError:
            pass
        return name, spawn_custom_bot(template(name)).run(scn).summary()

    with ThreadPoolExecutor(threads) as ex:
        for name, summary in ex.map(work, jobs):
            assert summary == serial[name]


@pytest.mark.parametrize("threads", [4, 16])
def test_shared_scenario_is_isolated_under_contention(threads, scn):
    before = fingerprint(scn)
    bots = [spawn_custom_bot(template(n)) for n in sorted(TEMPLATES)]

    def work(i):
        return bots[i % len(bots)].run(scn).net_profit_base

    with ThreadPoolExecutor(threads) as ex:
        got = list(ex.map(work, range(threads * 3)))
    expected = [bots[i % len(bots)].run(scn).net_profit_base for i in range(threads * 3)]
    assert got == expected and fingerprint(scn) == before


@pytest.mark.parametrize("threads", [2, 8])
def test_one_bot_instance_shared_between_threads(threads, scn, contexts):
    bot = spawn_custom_bot(template("transient_sandwich"))
    serial = [bot.plan(scn, c).net_profit_base for c in contexts]
    with ThreadPoolExecutor(threads) as ex:
        par = list(ex.map(lambda c: bot.plan(scn, c).net_profit_base, contexts))
    assert par == serial


def test_concurrent_overrides_do_not_leak(scn):
    cls = compile_blueprint(template("classic_sandwich"))

    def work(i):
        bot = cls({"min_profit_base": 10**30} if i % 2 else None)
        return i % 2, len(bot.run(scn).valid)

    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(work, range(16)))
    assert all(n == 0 for odd, n in rows if odd)
    assert len({n for odd, n in rows if not odd}) == 1
    assert cls.PIPELINE.get("min_profit_base") == 1


def test_scenario_cache_is_thread_safe():
    with ThreadPoolExecutor(8) as ex:
        scns = list(ex.map(lambda _: scenario_for(4242, 6), range(16)))
    assert all(s is scns[0] for s in scns)


def test_concurrent_transient_storage_is_per_run(scn, contexts):
    ctx, _ = first_sandwichable(scn, contexts)
    script = [{"op": "tstore", "key": "n", "value": {"victim_amount": True}},
              {"op": "if", "cond": {"tload": "n", "op": "==", "value": ctx.tx.amount_in}, "then": SANDWICH,
               "else": [{"op": "burn"}]}]
    bot = spawn_custom_bot(bp(script, name="tsiso", pipeline={"min_profit_base": 0, "require_flat": False}))
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda _: bot.execute(scn, ctx).result.ledger, range(24)))
    assert all(o == outs[0] for o in outs)


def test_worker_pool_evaluation_matches_inline():
    synth = BlueprintSynthesizer(bots=8, generations=1, seed=3, batch=2, scenario_seeds=1, victims=6)
    _, _, bps = synth.design(0)
    inline, tasks = synth.evaluate(bps)
    with WorkerPool(workers=2, parallel_threshold=1) as wp:
        par, _ = synth.evaluate(bps, wp)
    assert par == inline and tasks == 4


# ---------------------------------------------------------------------------
# Genes -> blueprints and AI synthesis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_random_genes_decode_to_valid_blueprints(seed):
    genes = np.random.default_rng(seed).random(N_BLUEPRINT_GENES)
    raw = blueprint_from_genes(genes, f"g{seed}")
    assert validate_blueprint(raw) == []
    assert compile_blueprint(raw).TRIGGER_CLASS == trigger_class(normalize_blueprint(raw)["trigger"])


@pytest.mark.parametrize("value", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_constant_genes_decode(value):
    assert validate_blueprint(blueprint_from_genes([value] * N_BLUEPRINT_GENES)) == []


def test_non_finite_genes_are_neutralised():
    genes = [float("nan")] * N_BLUEPRINT_GENES
    assert validate_blueprint(blueprint_from_genes(genes)) == []


def test_wrong_gene_count():
    with pytest.raises(BlueprintError):
        blueprint_from_genes([0.5] * (N_BLUEPRINT_GENES - 1))


def test_genes_reach_every_core_kind():
    kinds = set()
    for i, _ in enumerate(cb.CORE_KINDS):
        g = [0.4] * N_BLUEPRINT_GENES
        g[12] = (i + 0.5) / len(cb.CORE_KINDS)
        kinds.add(blueprint_from_genes(g)["description"])
    assert len(kinds) == len(cb.CORE_KINDS)


def test_genes_build_bridges_and_atomic_guards():
    g = [0.9] * N_BLUEPRINT_GENES
    g[12] = 0.1
    raw = blueprint_from_genes(g)
    guard = raw["script"][-1]
    assert guard["op"] == "atomic" and any(a["op"] == "bridge" for a in guard["body"])


def test_synthesizer_is_deterministic():
    a = BlueprintSynthesizer(bots=8, generations=2, seed=11, victims=6, scenario_seeds=1).run().summary()
    b = BlueprintSynthesizer(bots=8, generations=2, seed=11, victims=6, scenario_seeds=1).run().summary()
    assert a == b


def test_synthesizer_report(tmp_path):
    rep = BlueprintSynthesizer(bots=12, generations=2, seed=2, victims=6, scenario_seeds=1,
                               features=[0.2, -0.1]).run().summary()
    assert rep["invariants"]["ok"] and rep["status"] == "experimental"
    assert len(rep["generations"]) == 2 and rep["tasks"] == 6
    assert rep["best"]["blueprint"]["name"].startswith("ai-custom-g")
    for e in rep["novel_trigger_classes"]:
        assert e["trigger_class"] not in KNOWN_TRIGGER_CLASSES
    classes = [e["trigger_class"] for e in rep["novel_trigger_classes"]]
    assert len(classes) == len(set(classes)) == rep["novel_trigger_class_count"]
    json.dumps(rep)


def test_synthesizer_learns_from_its_elite():
    synth = BlueprintSynthesizer(bots=12, generations=2, seed=4, victims=6, scenario_seeds=1)
    before = synth.engine.flat().copy()
    rep = synth.run().summary()
    assert not np.allclose(before, synth.engine.flat())
    assert all(g["loss_after"] <= g["loss_before"] + 1e-9 for g in rep["generations"])


@pytest.mark.parametrize("kw", [dict(bots=0), dict(generations=0), dict(batch=0), dict(latent_dim=0),
                                dict(elite_frac=0), dict(features=[float("inf")])])
def test_synthesizer_rejects_bad_settings(kw):
    with pytest.raises(ConfigError):
        BlueprintSynthesizer(**kw)


def test_agent_factory_synthesizes_custom_bots():
    factory = AgentFactory(bots=4, generations=1, seed=1)
    rep = factory.synthesize_custom_bots(8, 1, victims=6, scenario_seeds=1)
    assert rep["invariants"]["ok"] and rep["bots_per_generation"] == 8
    assert rep["conditioned_on"]["regime"] == factory.metrics.regime
    assert len(rep["features"]) == len(rep["conditioned_on"]["features"])


def test_factory_features_condition_the_designs():
    a = AgentFactory(bots=4, generations=1, seed=1, regime="calm")
    b = AgentFactory(bots=4, generations=1, seed=1, regime="extreme")
    ra = a.synthesize_custom_bots(6, 1, victims=6, scenario_seeds=1)
    rb = b.synthesize_custom_bots(6, 1, victims=6, scenario_seeds=1)
    assert ra["features"] != rb["features"]


# ---------------------------------------------------------------------------
# Config, controller and CLI integration
# ---------------------------------------------------------------------------


def test_presets():
    assert SimulationConfig.for_mode("moderate", cpus=4).custom_bots == 0
    assert SimulationConfig.for_mode("brutal", cpus=4).custom_bots == 0
    mx = SimulationConfig.for_mode("max", cpus=4)
    assert mx.custom_bots > 0 and mx.custom_generations >= 1
    assert mx.to_dict()["custom_bots"] == mx.custom_bots


@pytest.mark.parametrize("kw", [dict(custom_bots=-1), dict(custom_generations=0), dict(custom_bots=10**9)])
def test_config_rejects_bad_custom_sizes(kw):
    with pytest.raises(ConfigError):
        SimulationConfig.for_mode("max", cpus=4, **kw)


def test_controller_section():
    cfg = SimulationConfig.for_mode("max", cpus=1, custom_bots=6, custom_generations=1)
    rep = run_custom_builder(cfg, None, {})
    assert rep["invariants"]["ok"] and rep["bots_per_generation"] == 6


def run_cli(capsys, *args):
    code = main(list(args))
    return code, capsys.readouterr().out


def test_cli_template(capsys):
    code, text = run_cli(capsys, "--bot-template", "classic_sandwich", "--seed", "2")
    data = json.loads(text)
    assert code == 0 and data["run"]["name"] == "classic_sandwich" and data["bytecode"]["script"]


def test_cli_schema_and_templates(capsys):
    code, text = run_cli(capsys, "--blueprint-schema")
    assert code == 0 and json.loads(text)["schema"] == 1
    code, text = run_cli(capsys, "--bot-templates")
    assert code == 0 and set(json.loads(text)) == set(TEMPLATES)


def test_cli_build_bot_file(tmp_path, capsys):
    path = tmp_path / "bot.json"
    path.write_text(json.dumps(template("transient_sandwich")), encoding="utf-8")
    code, text = run_cli(capsys, "--build-bot", str(path), "--scenario-victims", "6")
    assert code == 0 and json.loads(text)["blueprint"]["name"] == "transient_sandwich"


def test_cli_rejects_bad_blueprint(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"name": "x"}), encoding="utf-8")
    assert main(["--build-bot", str(path)]) == 2
