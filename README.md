# Knowledge Graph

Build a knowledge graph from documents, prediction markets, and live financial instruments — then explore and run predict-mode simulations in the browser.

## Quick start

```bash
cp .env.example .env          # fill in Azure OpenAI, Mistral OCR, AWS S3 keys
mkdir -p notes
cp your-documents/*.pdf notes/

source venv/bin/activate
pip install -r requirements.txt

python main.py                # full pipeline: index → sync → link → export
python viz/server.py          # open http://127.0.0.1:8765
```

## What `python main.py` does

One command runs the full pipeline in order:

1. **Index** — OCR documents in `notes/`, extract objects, states, and document edges
2. **Sync instruments** — refresh live prices from `instruments.yaml` (yfinance)
3. **Sync Polymarket** — fetch relevant prediction markets
4. **Link** — GPT edges: instruments→objects, markets→objects, markets→markets
5. **Export** — write `viz/graph.json` for the viewer

Skip phases via `.env`:

```
PIPELINE_POLYMARKET=0   # skip Polymarket
PIPELINE_INSTRUMENTS=0  # skip instruments
PIPELINE_LINK=0         # skip GPT linking
PIPELINE_EXPORT=0       # skip viz export
```

## Partial runs

```bash
python main.py index    # documents only
python main.py sync     # instruments + Polymarket
python main.py link     # all GPT linking passes
python main.py export   # re-export graph.json
```

## Viewer and agent

```bash
python viz/server.py
```

Open http://127.0.0.1:8765 — force-directed graph with an **Agent** chat panel:

- Ask questions → **predict mode** (ephemeral state propagation through the graph)
- `/sync polymarket`, `/sync instruments`, `/link all`, `/build`, `/help`

CLI predict mode (no browser):

```bash
python agent.py "What if Iran blockades the Strait of Hormuz?"
```

## Legacy scripts

`sync_polymarket.py`, `sync_instruments.py`, and `link_*.py` still work but print a deprecation notice and delegate to `pipeline.py`. Prefer `python main.py`.
