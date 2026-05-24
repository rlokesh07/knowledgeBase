# Knowledge Graph

Build a knowledge graph from documents, prediction markets, and live financial instruments — then explore it and run predict-mode simulations in the browser.

The pipeline OCRs your PDF/document collection, extracts named entities as graph nodes with embeddings, pulls live market data (Polymarket + yfinance), links everything with GPT-classified edges, and exports an interactive force-directed graph viewer with an AI chat agent.

## Architecture

```
notes/ (PDFs)
    │
    ▼
[Mistral OCR] → text chunks → [Azure OpenAI embeddings]
                                        │
                                        ▼
                              [GPT object extraction]
                              → Node(label, type, state)
                                        │
              ┌─────────────────────────┼──────────────────────┐
              ▼                         ▼                        ▼
    [Polymarket sync]         [yfinance instruments]   [GPT edge linking]
    → Node(market)            → Node(instrument)       → Edge(relationship)
              └─────────────────────────┴──────────────────────┘
                                        │
                                        ▼
                                 database.pkl
                                        │
                                        ▼
                               viz/graph.json
                               → browser viewer
                               → agent chat (predict mode)
```

**Node types:** `object` (entities from documents), `market` (Polymarket), `instrument` (financial)

**Edge types:** object↔object (topic similarity + GPT), instrument→object (GPT), market→object (GPT), market→market (implication, GPT)

## Quick start

```bash
cp .env.example .env          # fill in credentials (see below)
mkdir -p notes
cp your-documents/*.pdf notes/

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python main.py                # full pipeline: index → sync → link → export
python viz/server.py          # open http://127.0.0.1:8765
```

## Setup

### Required credentials

| Variable | Purpose |
|---|---|
| `AZURE_OPENAI_API_KEY` | Chat + embeddings |
| `AZURE_OPENAI_ENDPOINT` | Resource base URL (`https://<resource>.cognitiveservices.azure.com`) |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Deployment name for chat model |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Deployment name for embeddings (e.g. `text-embedding-3-small`) |
| `AZURE_API_KEY` | Mistral Document AI OCR key |
| `MISTRAL_OCR_ENDPOINT` | Full OCR URL from Azure deployment page |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | S3 upload (for OCR input) |
| `S3_BUCKET` | Bucket name (default: `srain`) |

### Optional

```env
FIRECRAWL_API_KEY=        # web ingestion (web_ingest.py)
NOTES_DIR=notes           # path to documents (default: notes/)
```

Copy `.env.example` to `.env` — it documents every tuneable parameter.

## Pipeline

```bash
python main.py            # all phases
python main.py index      # OCR + extract objects + document edges
python main.py sync       # instruments + Polymarket
python main.py link       # GPT edge linking passes
python main.py export     # re-export viz/graph.json
```

### Phases

1. **Index** — OCRs files in `notes/` (Mistral Document AI), splits into overlapping chunks, embeds (Azure OpenAI), extracts named entities as nodes, computes current state per node, builds object↔object edges.
2. **Sync instruments** — reads `instruments.yaml`, fetches live prices via yfinance, upserts instrument nodes.
3. **Sync Polymarket** — searches for relevant prediction markets, upserts market nodes.
4. **Link** — GPT edge passes: instrument→object, market→object, market→market implication.
5. **Export** — writes `viz/graph.json` for the browser viewer.

Skip phases:
```env
PIPELINE_INSTRUMENTS=0
PIPELINE_POLYMARKET=0
PIPELINE_LINK=0
PIPELINE_EXPORT=0
```

## Financial instruments

Configured in `instruments.yaml`:

```yaml
instruments:
  - id: sp500
    symbol: ^GSPC
    name: S&P 500
    asset_class: index
```

Add any yfinance-supported ticker. Asset classes: `fx`, `commodity`, `index`, `bond`, `equity`.

## Browser viewer

```bash
python viz/server.py
```

Opens http://127.0.0.1:8765 — force-directed graph with an **Agent** chat panel.

### Chat commands

| Command | Effect |
|---|---|
| Any question | Predict-mode simulation |
| `/sync polymarket` | Pull latest prediction markets |
| `/sync instruments` | Refresh live prices |
| `/link all` | Re-run GPT edge linking |
| `/build` | Full pipeline |
| `/help` | List commands |

## Predict mode

Predict mode answers questions by propagating an ephemeral state change through the graph. Given a question (e.g. *"What if Iran blockades the Strait of Hormuz?"*):

1. Embed the question and find seed nodes by cosine similarity
2. GPT scores how each seed node is affected
3. Propagate effects outward through edges (up to `PREDICT_MAX_DEPTH` hops)
4. GPT synthesizes a final report from all affected nodes

```bash
# CLI (no browser)
python agent.py "What if Iran blockades the Strait of Hormuz?"
```

Tuning:
```env
PREDICT_SEED_CANDIDATES=20
PREDICT_MAX_DEPTH=4
PREDICT_MAX_CHANGES=40
PREDICT_GPT_WORKERS=4
```

## Performance tuning

All defaults are conservative. Key knobs in `.env`:

| Variable | Default | Effect |
|---|---|---|
| `OBJECT_EXTRACT_BATCH_SIZE` | 20 | Chunks per GPT extraction call |
| `OBJECT_ASSIGN_MIN_COSINE` | 0.45 | Chunk→node similarity threshold |
| `EDGE_TOP_NODES_PER_CHUNK` | 6 | Candidate node pairs per chunk |
| `EDGE_MAX_PAIRS` | 300 | Max pairs sent to GPT |
| `EDGE_GPT_WORKERS` | 4 | Parallel GPT threads for edges |
| `STATE_GPT_BATCH_SIZE` | 4 | Objects per state-extraction call |
| `TEXT_CHUNK_MAX_CHARS` | 900 | Chunk size after OCR |
| `OBJECT_FULL_REFRESH` | off | Re-extract all objects from scratch |

## Data files

| File | Contents |
|---|---|
| `database.pkl` | Graph store (nodes, edges, chunks, embeddings) |
| `chunks/chunks.jsonl` | Chunk text + embeddings snapshot |
| `indexed_sources.json` | Paths of already-indexed documents |
| `disabled_tags.json` | Tags whose nodes are hidden from the graph |
| `instruments.yaml` | Financial instrument definitions |
| `viz/graph.json` | Exported graph for the browser viewer |
