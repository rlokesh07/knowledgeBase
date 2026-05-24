"""Ephemeral state-change propagation through the knowledge graph (predict mode)."""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

import graphStore
import aiEngine
from gpt_parallel import gpt_worker_count, run_parallel_batches
from pipeline import format_duration


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _predict_workers() -> int:
    raw = (os.environ.get("PREDICT_GPT_WORKERS") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return gpt_worker_count()


@dataclass
class _EvalWork:
    target_uuid: str
    from_uuid: str
    incoming: str
    depth: int


@dataclass
class _EvalResult:
    work: _EvalWork
    prev_state: str
    eval_out: object | None
    error: str = ""


@dataclass
class PropagationChange:
    node_uuid: str
    label: str
    object_type: str
    depth: int
    from_label: str
    incoming_change: str
    previous_state: str
    new_state: str
    reasoning: str
    is_live_signal: bool = False


@dataclass
class PropagationResult:
    seed_uuid: str
    seed_label: str
    seed_reasoning: str
    changes: list[PropagationChange] = field(default_factory=list)
    answer: str = ""


def _baseline_state(node: graphStore.Node) -> str:
    return (getattr(node, "state", "") or "").strip()


def _is_live_signal(node: graphStore.Node) -> bool:
    return (getattr(node, "object_type", "") or "") in {"market", "instrument"}


def _active_nodes(store: graphStore.GraphStore) -> list[graphStore.Node]:
    return [n for n in store.nodes if not getattr(n, "disabled", False)]


def _undirected_neighbors(
    store: graphStore.GraphStore,
    node_uuid: str,
) -> list[tuple[str, str, str]]:
    """Return (neighbor_uuid, relationship_type, mechanism) for all attached edges."""
    out: list[tuple[str, str, str]] = []
    for edge in store.edges:
        mech = (getattr(edge, "mechanism", "") or "").strip()
        rtype = (getattr(edge, "relationship_type", "") or "").strip()
        if edge.uuid1 == node_uuid:
            out.append((edge.uuid2, rtype, mech))
        elif edge.uuid2 == node_uuid:
            out.append((edge.uuid1, rtype, mech))
    return out


def _edge_mechanism(
    store: graphStore.GraphStore,
    uuid_a: str,
    uuid_b: str,
) -> str:
    for edge in store.edges:
        if {edge.uuid1, edge.uuid2} == {uuid_a, uuid_b}:
            return (getattr(edge, "mechanism", "") or "").strip()
    return ""


def _node_payload(
    node: graphStore.Node,
    simulated: dict[str, str],
) -> dict:
    state = simulated.get(node.uuid, _baseline_state(node))
    props = getattr(node, "properties", None) or {}
    payload = {
        "uuid": node.uuid,
        "label": node.label,
        "object_type": node.object_type,
        "description": (node.description or "")[:600],
        "state": state[:1200],
    }
    if node.object_type == "instrument":
        payload["symbol"] = props.get("symbol") or ""
    if node.object_type == "market":
        payload["outcome_prices"] = props.get("outcome_prices") or []
    return payload


def run_propagation(
    store: graphStore.GraphStore,
    engine: aiEngine.aiEngine,
    question: str,
    candidates: list[graphStore.Node],
    progress,
) -> PropagationResult | None:
    """Run ephemeral predict-mode propagation. Does not mutate node.state on store."""
    run_start = time.perf_counter()
    if not candidates:
        progress("No candidate nodes for predict mode.")
        return None

    node_by_uuid = {n.uuid: n for n in _active_nodes(store)}
    cand_payload = [_node_payload(n, {}) for n in candidates if n.uuid in node_by_uuid]

    progress("Predict mode: selecting seed node and scenario change…")
    seed_out = engine.select_seed_change(question, cand_payload)
    if seed_out is None:
        progress("Seed selection failed.")
        return None

    seed = node_by_uuid.get(seed_out.seed_uuid)
    if seed is None:
        progress("Seed uuid not found in graph.")
        return None

    max_depth = _env_int("PREDICT_MAX_DEPTH", 4)
    max_changes = _env_int("PREDICT_MAX_CHANGES", 40)

    simulated: dict[str, str] = {}
    changes: list[PropagationChange] = []
    visited: set[str] = set()

    prev_seed = _baseline_state(seed)
    simulated[seed.uuid] = seed_out.new_state.strip()
    visited.add(seed.uuid)

    changes.append(
        PropagationChange(
            node_uuid=seed.uuid,
            label=seed.label,
            object_type=seed.object_type or "other",
            depth=0,
            from_label="(scenario)",
            incoming_change=seed_out.hypothetical_change,
            previous_state=prev_seed,
            new_state=simulated[seed.uuid],
            reasoning=seed_out.reasoning,
            is_live_signal=_is_live_signal(seed),
        )
    )
    progress(
        f"Seed: {seed.label} — {seed_out.hypothetical_change[:120]}"
    )

    pending_by_depth: dict[int, list[_EvalWork]] = defaultdict(list)
    for neighbor_uuid, _rtype, _mech in _undirected_neighbors(store, seed.uuid):
        if neighbor_uuid not in visited:
            pending_by_depth[1].append(
                _EvalWork(neighbor_uuid, seed.uuid, seed_out.hypothetical_change, 1)
            )

    workers = _predict_workers()
    merge_lock = threading.Lock()

    for depth in range(1, max_depth + 1):
        if len(changes) >= max_changes:
            break

        raw_events = pending_by_depth.pop(depth, [])
        if not raw_events:
            continue

        to_eval: list[_EvalWork] = []
        seen_at_depth: set[str] = set()
        for work in raw_events:
            if work.target_uuid in visited or work.target_uuid in seen_at_depth:
                continue
            seen_at_depth.add(work.target_uuid)
            visited.add(work.target_uuid)
            to_eval.append(work)

        if not to_eval:
            continue

        n_workers = max(1, min(workers, len(to_eval)))
        progress(
            f"  Depth {depth}: evaluating {len(to_eval)} node(s)"
            + (f" ({n_workers} worker thread(s))…" if n_workers > 1 else "…")
        )

        # Snapshot simulated state for this depth — all evaluations read the same prior state.
        simulated_snapshot = dict(simulated)
        depth_results: list[_EvalResult] = []

        def _eval_one(_batch_idx: int, work: _EvalWork) -> list[_EvalResult]:
            target = node_by_uuid.get(work.target_uuid)
            from_node = node_by_uuid.get(work.from_uuid)
            if target is None or from_node is None:
                return [
                    _EvalResult(work, "", None, error="missing node")
                ]

            prev_state = simulated_snapshot.get(work.target_uuid, _baseline_state(target))
            mechanism = _edge_mechanism(store, work.from_uuid, work.target_uuid)
            try:
                eval_out = engine.evaluate_propagated_change(
                    _node_payload(target, simulated_snapshot),
                    incoming_change=work.incoming,
                    from_label=from_node.label,
                    edge_mechanism=mechanism,
                )
            except Exception as exc:
                return [_EvalResult(work, prev_state, None, error=str(exc))]
            return [_EvalResult(work, prev_state, eval_out)]

        def _merge_results(batch: list[_EvalResult]) -> None:
            depth_results.extend(batch)

        run_parallel_batches(
            to_eval,
            worker=_eval_one,
            on_results=_merge_results,
            progress=progress,
            label=f"Predict depth {depth}",
            workers=n_workers,
            lock=merge_lock,
        )

        for result in depth_results:
            work = result.work
            target = node_by_uuid.get(work.target_uuid)
            if target is None:
                continue

            if result.error:
                progress(f"    Error [{depth}]: {target.label} — {result.error[:80]}")
                continue

            eval_out = result.eval_out
            if eval_out is None:
                progress(f"    Skipped (content filter): {target.label}")
                continue
            if not getattr(eval_out, "significant", False) or not (
                getattr(eval_out, "new_state", "") or ""
            ).strip():
                progress(f"    Not significant: {target.label}")
                continue

            new_state = eval_out.new_state.strip()
            simulated[work.target_uuid] = new_state
            live = _is_live_signal(target)
            changes.append(
                PropagationChange(
                    node_uuid=work.target_uuid,
                    label=target.label,
                    object_type=target.object_type or "other",
                    depth=depth,
                    from_label=node_by_uuid[work.from_uuid].label,
                    incoming_change=work.incoming,
                    previous_state=result.prev_state,
                    new_state=new_state,
                    reasoning=(getattr(eval_out, "reasoning", "") or "").strip(),
                    is_live_signal=live,
                )
            )

            tag = " [live signal]" if live else ""
            progress(f"    Significant{tag}: {target.label}")

            if len(changes) >= max_changes or depth >= max_depth:
                continue

            downstream = (
                getattr(eval_out, "change_to_propagate", "") or work.incoming
            ).strip()
            for neighbor_uuid, _rtype, _mech in _undirected_neighbors(store, work.target_uuid):
                if neighbor_uuid not in visited:
                    pending_by_depth[depth + 1].append(
                        _EvalWork(neighbor_uuid, work.target_uuid, downstream, depth + 1)
                    )

    seed_info = {
        "label": seed.label,
        "object_type": seed.object_type,
        "hypothetical_change": seed_out.hypothetical_change,
        "new_state": simulated[seed.uuid],
        "reasoning": seed_out.reasoning,
    }
    change_payload = [
        {
            "depth": c.depth,
            "label": c.label,
            "object_type": c.object_type,
            "from_label": c.from_label,
            "incoming_change": c.incoming_change,
            "previous_state": c.previous_state[:600],
            "new_state": c.new_state[:800],
            "reasoning": c.reasoning,
            "is_live_signal": c.is_live_signal,
        }
        for c in changes
    ]

    progress(f"Propagation done: {len(changes)} significant change(s). Synthesizing report…")
    report = engine.synthesize_propagation_report(question, seed_info, change_payload)
    answer = report.answer if report else _fallback_report(question, changes)

    elapsed = time.perf_counter() - run_start
    progress(f"--- Predict mode: {format_duration(elapsed)} ---")

    return PropagationResult(
        seed_uuid=seed.uuid,
        seed_label=seed.label,
        seed_reasoning=seed_out.reasoning,
        changes=changes,
        answer=answer,
    )


def _fallback_report(question: str, changes: list[PropagationChange]) -> str:
    lines = [f"Question: {question}", "", "Significant changes:"]
    for c in changes:
        prefix = "[market/instrument] " if c.is_live_signal else "• "
        lines.append(
            f"{prefix}[depth {c.depth}] {c.label} ({c.object_type}): {c.new_state[:400]}"
        )
    return "\n".join(lines)
