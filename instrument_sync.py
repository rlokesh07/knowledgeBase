"""Sync financial instrument quotes into the knowledge graph as instrument nodes."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import graphStore
from instrument_client import InstrumentQuote, InstrumentSpec, InstrumentProvider, YFinanceProvider

TOPIC_UUID_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "kms.topic-node.v1")


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def instruments_config_path(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parent
    raw = (os.environ.get("INSTRUMENTS_CONFIG") or "instruments.yaml").strip()
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path


def load_instrument_specs(path: Path | None = None) -> list[InstrumentSpec]:
    config_path = path or instruments_config_path()
    if not config_path.exists():
        return []

    text = config_path.read_text(encoding="utf-8")
    rows: list[dict] = []

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit(
                "PyYAML is required to load instruments.yaml — pip install pyyaml"
            ) from exc
        data = yaml.safe_load(text) or {}
        rows = data.get("instruments") or []
    else:
        import json

        data = json.loads(text)
        rows = data.get("instruments") or []

    specs: list[InstrumentSpec] = []
    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        inst_id = str(row.get("id") or "").strip()
        symbol = str(row.get("symbol") or "").strip()
        name = str(row.get("name") or symbol or inst_id).strip()
        asset_class = str(row.get("asset_class") or "other").strip()
        if not inst_id or not symbol or inst_id in seen_ids:
            continue
        seen_ids.add(inst_id)
        specs.append(
            InstrumentSpec(
                id=inst_id,
                symbol=symbol,
                name=name,
                asset_class=asset_class,
            )
        )
    return specs


def append_to_registry(path: Path, specs: list[InstrumentSpec]) -> int:
    """Persist new instrument specs to the YAML registry. Returns count appended."""
    if not specs:
        return 0

    existing = load_instrument_specs(path)
    existing_ids = {s.id for s in existing}
    existing_symbols = {s.symbol.upper() for s in existing}
    to_add = [
        s
        for s in specs
        if s.id not in existing_ids and s.symbol.upper() not in existing_symbols
    ]
    if not to_add:
        return 0

    try:
        import yaml
    except ImportError:
        return 0

    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        data = {}
    instruments = list(data.get("instruments") or [])
    for spec in to_add:
        instruments.append(
            {
                "id": spec.id,
                "symbol": spec.symbol,
                "name": spec.name,
                "asset_class": spec.asset_class,
            }
        )
    data["instruments"] = instruments
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return len(to_add)


def instrument_node_uuid(instrument_id: str) -> str:
    return str(uuid.uuid5(TOPIC_UUID_NS, f"instrument:{instrument_id}"))


def instrument_catalog(store: graphStore.GraphStore) -> list[dict]:
    out: list[dict] = []
    for node in store.nodes:
        if (getattr(node, "object_type", "") or "") != "instrument":
            continue
        if getattr(node, "disabled", False):
            continue
        props = getattr(node, "properties", None) or {}
        out.append(
            {
                "uuid": node.uuid,
                "id": props.get("instrument_id") or "",
                "symbol": props.get("symbol") or "",
                "name": node.label,
                "asset_class": props.get("asset_class") or "",
                "state": (getattr(node, "state", "") or "")[:400],
            }
        )
    return out


def _instrument_description(quote: InstrumentQuote) -> str:
    return (
        f"Live market instrument ({quote.asset_class}): {quote.name} "
        f"tracked via {quote.symbol}."
    )


def _instrument_properties(quote: InstrumentQuote) -> dict:
    return {
        "instrument_id": quote.instrument_id,
        "symbol": quote.symbol,
        "asset_class": quote.asset_class,
        "price": quote.price,
        "currency": quote.currency,
        "change_1d_pct": quote.change_1d_pct,
        "change_1w_pct": quote.change_1w_pct,
        "as_of": quote.as_of,
        "source": "yfinance",
    }


def _upsert_instrument_node(
    store: graphStore.GraphStore,
    quote: InstrumentQuote,
    *,
    next_cluster_id: list[int],
) -> tuple[graphStore.Node, bool]:
    """Insert or update an instrument node. Returns (node, created)."""
    nid = instrument_node_uuid(quote.instrument_id)
    state = quote.format_state()
    props = _instrument_properties(quote)

    for node in store.nodes:
        props_id = (node.properties or {}).get("instrument_id")
        props_symbol = (node.properties or {}).get("symbol")
        if (
            node.uuid == nid
            or props_id == quote.instrument_id
            or (props_symbol and str(props_symbol).upper() == quote.symbol.upper())
        ):
            node.label = quote.name
            node.object_type = "instrument"
            node.description = _instrument_description(quote)
            node.state = state
            node.properties = props
            return node, False

    node = graphStore.Node(
        nid,
        quote.name,
        [],
        cluster_id=next_cluster_id[0],
        object_type="instrument",
        description=_instrument_description(quote),
        state=state,
        properties=props,
        tags=["instrument"],
    )
    store.nodes.append(node)
    next_cluster_id[0] += 1
    return node, True


def spec_from_suggestion(row: dict) -> InstrumentSpec | None:
    inst_id = str(row.get("id") or "").strip()
    symbol = str(row.get("symbol") or "").strip()
    name = str(row.get("name") or symbol or inst_id).strip()
    asset_class = str(row.get("asset_class") or "other").strip()
    if not inst_id or not symbol:
        return None
    return InstrumentSpec(id=inst_id, symbol=symbol, name=name, asset_class=asset_class)


def ensure_instruments(
    store: graphStore.GraphStore,
    provider: InstrumentProvider,
    specs: list[InstrumentSpec],
    progress,
) -> tuple[list[InstrumentQuote], list[graphStore.Node]]:
    """Fetch quotes, upsert nodes for valid symbols. Returns (quotes, touched_nodes)."""
    if not specs:
        return [], []

    progress(f"Fetching quotes for {len(specs)} instrument(s)…")
    quotes = provider.fetch_quotes(specs)
    if not quotes:
        progress("No valid quotes returned.")
        return [], []

    fetched_ids = {q.instrument_id for q in quotes}
    for spec in specs:
        if spec.id not in fetched_ids:
            progress(f"  Skipped (no quote): {spec.symbol} ({spec.name})")

    next_id = max((n.cluster_id for n in store.nodes), default=-1) + 1
    counter = [next_id]
    touched: list[graphStore.Node] = []

    for quote in quotes:
        node, _created = _upsert_instrument_node(store, quote, next_cluster_id=counter)
        touched.append(node)
        progress(f"  Updated: {quote.name} ({quote.symbol}) — {quote.format_state()[:120]}")

    return quotes, touched


def sync_instruments_from_config(
    store: graphStore.GraphStore,
    progress,
    *,
    config_path: Path | None = None,
    provider: InstrumentProvider | None = None,
) -> tuple[int, int]:
    """Load YAML watchlist, fetch quotes, upsert nodes. Returns (added, updated)."""
    specs = load_instrument_specs(config_path)
    if not specs:
        progress("No instruments in registry.")
        return 0, 0

    prov = provider or YFinanceProvider()
    before_uuids = {n.uuid for n in store.nodes}
    quotes, touched = ensure_instruments(store, prov, specs, progress)
    added = sum(1 for n in touched if n.uuid not in before_uuids)
    updated = len(touched) - added
    progress(f"Instrument sync done: {added} added, {updated} updated ({len(quotes)} quoted).")
    return added, updated
