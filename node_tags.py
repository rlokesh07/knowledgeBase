"""Node topical tags and global disabled-tag state."""

import json
import re
from pathlib import Path

import graphStore

ROOT = Path(__file__).resolve().parent
DISABLED_TAGS_PATH = ROOT / "disabled_tags.json"


def normalize_tag(tag: str) -> str:
    s = (tag or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_tags(tags: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in tags or []:
        t = normalize_tag(raw)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def load_disabled_tags() -> set[str]:
    if not DISABLED_TAGS_PATH.exists():
        return set()
    try:
        data = json.loads(DISABLED_TAGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    raw = data.get("disabled_tags") or []
    return {normalize_tag(t) for t in raw if normalize_tag(t)}


def save_disabled_tags(tags: set[str]) -> None:
    ordered = sorted(tags)
    DISABLED_TAGS_PATH.write_text(
        json.dumps({"disabled_tags": ordered}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def apply_disabled_tags(
    store: graphStore.GraphStore,
    disabled_tags: set[str] | None = None,
) -> None:
    disabled = disabled_tags if disabled_tags is not None else load_disabled_tags()
    for node in store.nodes:
        node_tags = {normalize_tag(t) for t in (getattr(node, "tags", None) or [])}
        node.disabled = bool(node_tags & disabled)


def sync_disabled_state(store: graphStore.GraphStore) -> None:
    apply_disabled_tags(store)


def set_disabled_tags(
    store: graphStore.GraphStore,
    disabled_tags: set[str],
    *,
    db_path: Path | None = None,
) -> None:
    normalized = {normalize_tag(t) for t in disabled_tags if normalize_tag(t)}
    save_disabled_tags(normalized)
    apply_disabled_tags(store, normalized)
    if db_path is not None:
        store.exportToFile(str(db_path))


def collect_tag_counts(store: graphStore.GraphStore) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in store.nodes:
        for tag in normalize_tags(getattr(node, "tags", None)):
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def tag_summary(store: graphStore.GraphStore) -> list[dict]:
    counts = collect_tag_counts(store)
    disabled = load_disabled_tags()
    return [
        {
            "tag": tag,
            "count": counts[tag],
            "disabled": tag in disabled,
        }
        for tag in sorted(counts)
    ]


def active_nodes(nodes: list[graphStore.Node]) -> list[graphStore.Node]:
    return [n for n in nodes if not getattr(n, "disabled", False)]
