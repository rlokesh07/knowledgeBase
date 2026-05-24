"""Fetch live quotes for financial instruments via Yahoo Finance (yfinance)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol


@dataclass
class InstrumentSpec:
    id: str
    symbol: str
    name: str
    asset_class: str


@dataclass
class InstrumentQuote:
    instrument_id: str
    symbol: str
    name: str
    price: float
    currency: str
    change_1d_pct: float | None
    change_1w_pct: float | None
    as_of: str
    asset_class: str

    def format_state(self) -> str:
        if self.asset_class == "bond":
            level = f"{self.price:.2f}%"
        else:
            level = f"{self.currency} {self.price:,.2f}".strip()

        parts = [f"{self.name} ({self.symbol}) at {level}"]
        if self.change_1d_pct is not None:
            parts.append(f"{self.change_1d_pct:+.1f}% today")
        if self.change_1w_pct is not None:
            parts.append(f"{self.change_1w_pct:+.1f}% over 7 days")
        parts.append(f"as of {self.as_of[:10]}")
        return ", ".join(parts) + "."


class InstrumentProvider(Protocol):
    def fetch_quotes(self, specs: list[InstrumentSpec]) -> list[InstrumentQuote]: ...


def _pct_change(current: float, previous: float | None) -> float | None:
    if previous is None or previous == 0:
        return None
    return (current - previous) / abs(previous) * 100.0


class YFinanceProvider:
    """Yahoo Finance adapter — no API key required."""

    def fetch_quotes(self, specs: list[InstrumentSpec]) -> list[InstrumentQuote]:
        if not specs:
            return []

        import yfinance as yf

        symbols = [s.symbol for s in specs]
        spec_by_symbol = {s.symbol: s for s in specs}
        quotes: list[InstrumentQuote] = []

        for symbol in symbols:
            spec = spec_by_symbol[symbol]
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="8d", auto_adjust=True)
                if hist.empty:
                    continue

                current = float(hist["Close"].iloc[-1])
                prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else None
                week_ago = float(hist["Close"].iloc[0]) if len(hist) >= 2 else None

                info = {}
                try:
                    info = ticker.fast_info  # type: ignore[attr-defined]
                except Exception:
                    pass

                currency = ""
                if hasattr(info, "currency"):
                    currency = str(getattr(info, "currency", "") or "")
                if not currency:
                    currency = str((ticker.info or {}).get("currency") or "USD")

                as_of_ts = hist.index[-1]
                if hasattr(as_of_ts, "to_pydatetime"):
                    as_of_dt = as_of_ts.to_pydatetime()
                else:
                    as_of_dt = datetime.now(timezone.utc)
                if as_of_dt.tzinfo is None:
                    as_of_dt = as_of_dt.replace(tzinfo=timezone.utc)

                quotes.append(
                    InstrumentQuote(
                        instrument_id=spec.id,
                        symbol=symbol,
                        name=spec.name,
                        price=current,
                        currency=currency,
                        change_1d_pct=_pct_change(current, prev_close),
                        change_1w_pct=_pct_change(current, week_ago),
                        as_of=as_of_dt.isoformat(),
                        asset_class=spec.asset_class,
                    )
                )
            except Exception:
                continue

        return quotes
