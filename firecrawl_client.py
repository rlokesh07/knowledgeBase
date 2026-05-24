"""Thin wrapper around Firecrawl search and scrape."""

import os
from dataclasses import dataclass, field


@dataclass
class WebDocument:
    url: str
    title: str
    description: str = ""
    markdown: str = ""
    category: str = ""


def _require_firecrawl():
    try:
        from firecrawl import Firecrawl
    except ImportError as exc:
        raise SystemExit(
            "firecrawl-py is not installed. Run: pip install firecrawl-py"
        ) from exc
    return Firecrawl


def _as_dict(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(getattr(obj, "__dict__", {}) or {})


def _pick_str(d: dict, *keys: str) -> str:
    for key in keys:
        val = d.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _normalize_hit(raw) -> WebDocument | None:
    d = _as_dict(raw)
    url = _pick_str(d, "url", "source_url", "sourceURL")
    if not url:
        meta = _as_dict(d.get("metadata"))
        url = _pick_str(meta, "url", "source_url", "sourceURL")
    if not url:
        return None
    title = _pick_str(d, "title") or _pick_str(_as_dict(d.get("metadata")), "title") or url
    return WebDocument(
        url=url,
        title=title,
        description=_pick_str(d, "description", "snippet"),
        markdown=_pick_str(d, "markdown"),
        category=_pick_str(d, "category"),
    )


class FirecrawlClient:
    def __init__(self, api_key: str | None = None):
        key = (api_key or os.environ.get("FIRECRAWL_API_KEY") or "").strip()
        if not key:
            raise SystemExit(
                "Missing FIRECRAWL_API_KEY. Set it in .env to enable web research."
            )
        Firecrawl = _require_firecrawl()
        self._client = Firecrawl(api_key=key)

    def search(self, query: str, *, limit: int = 5) -> list[WebDocument]:
        query = (query or "").strip()
        if not query:
            return []

        raw = self._client.search(
            query,
            limit=limit,
            scrape_options={"formats": ["markdown"]},
        )
        data = _as_dict(raw)
        if "data" in data:
            data = _as_dict(data["data"])

        hits: list[WebDocument] = []
        seen: set[str] = set()
        for bucket in ("web", "news"):
            items = data.get(bucket) or []
            if not isinstance(items, list):
                continue
            for item in items:
                doc = _normalize_hit(item)
                if doc and doc.url not in seen:
                    seen.add(doc.url)
                    hits.append(doc)
        return hits[:limit]

    def scrape(self, url: str) -> WebDocument:
        url = (url or "").strip()
        if not url:
            raise ValueError("url is required")

        raw = self._client.scrape(url, formats=["markdown"])
        d = _as_dict(raw)
        doc = _normalize_hit(d)
        if not doc:
            meta = _as_dict(d.get("metadata"))
            doc = WebDocument(
                url=_pick_str(meta, "url", "source_url", "sourceURL") or url,
                title=_pick_str(meta, "title") or url,
                markdown=_pick_str(d, "markdown"),
            )
        if not doc.markdown.strip():
            doc.markdown = _pick_str(d, "markdown")
        return doc
