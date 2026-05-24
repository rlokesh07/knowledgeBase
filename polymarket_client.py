"""Fetch prediction markets from the Polymarket Gamma API."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

_GAMMA_BASE = "https://gamma-api.polymarket.com"
_DEFAULT_TIMEOUT = 30


@dataclass
class PolymarketMarket:
    id: str
    question: str
    slug: str
    description: str
    outcomes: list[str]
    outcome_prices: list[float]
    volume: float
    liquidity: float
    end_date: str
    active: bool
    closed: bool
    event_title: str = ""
    event_slug: str = ""
    url: str = ""

    def format_state(self) -> str:
        """Human-readable state snapshot: current odds."""
        if not self.outcomes or not self.outcome_prices:
            return "Polymarket market with no current pricing data."

        parts = []
        for label, price in zip(self.outcomes, self.outcome_prices):
            pct = max(0.0, min(100.0, price * 100.0))
            parts.append(f"{label}: {pct:.1f}%")
        odds = " | ".join(parts)

        lines = [f"Polymarket prediction market odds — {odds}."]
        if self.volume:
            lines.append(f"Trading volume: ${self.volume:,.0f}.")
        if self.liquidity:
            lines.append(f"Liquidity: ${self.liquidity:,.0f}.")
        if self.end_date:
            lines.append(f"Resolution deadline: {self.end_date}.")
        if self.url:
            lines.append(f"Source: {self.url}")
        return " ".join(lines)


def _parse_json_list(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    return []


def _parse_prices(raw) -> list[float]:
    out: list[float] = []
    for item in _parse_json_list(raw):
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def _market_from_dict(d: dict, *, event_title: str = "", event_slug: str = "") -> PolymarketMarket | None:
    market_id = str(d.get("id") or "").strip()
    question = (d.get("question") or d.get("title") or "").strip()
    if not market_id or not question:
        return None

    slug = (d.get("slug") or "").strip()
    event_slug = event_slug or slug
    url = f"https://polymarket.com/event/{event_slug}" if event_slug else ""

    return PolymarketMarket(
        id=market_id,
        question=question,
        slug=slug,
        description=(d.get("description") or "").strip(),
        outcomes=[str(x) for x in _parse_json_list(d.get("outcomes"))],
        outcome_prices=_parse_prices(d.get("outcomePrices")),
        volume=float(d.get("volumeNum") or d.get("volume") or 0),
        liquidity=float(d.get("liquidityNum") or d.get("liquidity") or 0),
        end_date=(d.get("endDateIso") or d.get("endDate") or "")[:10],
        active=bool(d.get("active", True)),
        closed=bool(d.get("closed", False)),
        event_title=event_title,
        event_slug=event_slug,
        url=url,
    )


class PolymarketClient:
    def __init__(self, base_url: str = _GAMMA_BASE, timeout: int = _DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: dict | None = None) -> object:
        resp = requests.get(
            f"{self.base_url}{path}",
            params=params or {},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def search(self, query: str, *, limit: int = 20) -> list[PolymarketMarket]:
        query = (query or "").strip()
        if not query:
            return []

        data = self._get("/public-search", {"q": query, "limit": limit})
        events = data.get("events") if isinstance(data, dict) else []
        if not isinstance(events, list):
            return []

        markets: list[PolymarketMarket] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            event_title = (event.get("title") or "").strip()
            event_slug = (event.get("slug") or "").strip()
            nested = event.get("markets") or []
            if not isinstance(nested, list):
                continue
            for raw in nested:
                if not isinstance(raw, dict):
                    continue
                m = _market_from_dict(raw, event_title=event_title, event_slug=event_slug)
                if m and m.active and not m.closed:
                    markets.append(m)
        return markets

    def list_active_markets(self, *, limit: int = 100, offset: int = 0) -> list[PolymarketMarket]:
        data = self._get(
            "/markets",
            {
                "limit": limit,
                "offset": offset,
                "active": "true",
                "closed": "false",
            },
        )
        if not isinstance(data, list):
            return []
        out: list[PolymarketMarket] = []
        for raw in data:
            if isinstance(raw, dict):
                m = _market_from_dict(raw)
                if m and m.active and not m.closed:
                    out.append(m)
        return out


def search_terms_from_graph(nodes, *, max_terms: int = 25) -> list[str]:
    """Derive Polymarket search queries from ontology object nodes."""
    ranked = sorted(nodes, key=lambda n: len(n.chunk_uuids), reverse=True)
    terms: list[str] = []
    seen: set[str] = set()

    for node in ranked:
        if getattr(node, "disabled", False):
            continue
        label = (node.label or "").strip()
        if not label or len(label) > 70:
            continue
        otype = (getattr(node, "object_type", "") or "").lower()
        if otype not in ("person", "place", "organization", "event", "concept", "process"):
            continue
        if otype == "artifact" and len(label) > 35:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(label)
        if len(terms) >= max_terms:
            break
    return terms
