import os
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlparse, urlunparse

import boto3
from dotenv import load_dotenv

import aiEngine
import brain
import graphStore
from encoder import Encoder


def _progress(message: str) -> None:
    print(message, flush=True)


def _indexed_source_paths(store: graphStore.GraphStore) -> set[str]:
    """Resolved filesystem paths already represented by at least one graph node."""
    paths: set[str] = set()
    for node in store.nodes:
        raw = (node.link or "").strip()
        if not raw:
            continue
        try:
            paths.add(str(Path(raw).expanduser().resolve(strict=False)))
        except OSError:
            paths.add(raw)
    return paths


# Azure OpenAI REST API revision (fallback if URL / env omit api-version).
AZURE_OPENAI_HTTP_API_VERSION = "2024-02-15-preview"

_MISTRAL_OCR_ROUTE = "/providers/mistral/azure/ocr"


def _normalize_mistral_ocr_endpoint(raw: str) -> str:
    """Rewrite common mis-paste OpenAI Cognitive Services host → Foundry OCR URL.

    Runtime evidence showed POST to ``*.cognitiveservices.azure.com`` with no OCR path yields
    HTTP 200 ``Content-Length: 0``. Microsoft samples use ``<name>.services.ai.azure.com``.
    """
    s = raw.strip()
    if not s:
        return s
    u = urlparse(s)
    host = (u.hostname or "").lower()
    netloc_original = u.netloc
    query = u.query
    fragment = u.fragment
    scheme = u.scheme or "https"
    path = (u.path or "").rstrip("/")

    if _MISTRAL_OCR_ROUTE in path.lower():
        return urlunparse((scheme, netloc_original, path or "/", "", query, fragment))

    cog_suffix = ".cognitiveservices.azure.com"
    if host.endswith(cog_suffix):
        stem = host[: -len(cog_suffix)].rstrip(".")
        if stem:
            new_netloc = f"{stem}.services.ai.azure.com"
            rebuilt = urlunparse(
                (
                    scheme,
                    new_netloc,
                    _MISTRAL_OCR_ROUTE,
                    "",
                    query,
                    fragment,
                )
            )
            return rebuilt

    sia_suffix = ".services.ai.azure.com"
    if host.endswith(sia_suffix) and path in ("", "/"):
        return urlunparse(
            (scheme, netloc_original, _MISTRAL_OCR_ROUTE, "", query, fragment)
        )

    return s


def _normalize_azure_openai_endpoint(raw: str) -> tuple[str, str | None]:
    """Collapse a pasted portal URL to scheme://host only.

    If the URL includes ``api-version`` in its query string, returns that value
    (used when AZURE_OPENAI_API_VERSION is unset).
    """
    u = urlparse(raw.strip())
    if not u.scheme or not u.netloc:
        raise SystemExit(
            "AZURE_OPENAI_ENDPOINT must be a URL like "
            "`https://<resource>.cognitiveservices.azure.com` or "
            "`https://<resource>.openai.azure.com` "
            "(not an empty host)."
        )
    base_url = f"{u.scheme}://{u.netloc}"
    qp = parse_qs(u.query)
    api_from_url = (qp.get("api-version") or [None])[0]
    if isinstance(api_from_url, str):
        api_from_url = api_from_url.strip() or None
    return base_url, api_from_url


def _require_env(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and set them."
        )


def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    _progress("Indexing starting… loading environment")

    _require_env(
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "MISTRAL_OCR_ENDPOINT",
        "S3_BUCKET",
    )

    mistral_key = (
        (os.environ.get("AZURE_API_KEY") or os.environ.get("MISTRAL_API_KEY") or "").strip()
    )
    if not mistral_key:
        raise SystemExit(
            "Missing Mistral OCR API key: set AZURE_API_KEY (Azure samples) "
            "or MISTRAL_API_KEY in .env to the key from Deployments + Endpoint."
        )

    notes_dir = Path(os.environ.get("NOTES_DIR", "notes"))
    chunks_dir = Path(os.environ.get("CHUNKS_DIR", "chunks"))
    if not notes_dir.is_absolute():
        notes_dir = project_root / notes_dir
    if not chunks_dir.is_absolute():
        chunks_dir = project_root / chunks_dir
    chunks_dir.mkdir(parents=True, exist_ok=True)

    if notes_dir.exists() and not notes_dir.is_dir():
        raise SystemExit(f"NOTES_DIR exists but is not a directory: {notes_dir}")
    notes_dir.mkdir(parents=True, exist_ok=True)

    chat_deployment = (os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"] or "").strip()
    raw_aoai = (os.environ["AZURE_OPENAI_ENDPOINT"] or "").strip()
    azure_base, api_hint_from_url = _normalize_azure_openai_endpoint(raw_aoai)

    api_version = (
        (os.environ.get("AZURE_OPENAI_API_VERSION") or "").strip()
        or (api_hint_from_url or "").strip()
        or AZURE_OPENAI_HTTP_API_VERSION
    )

    engine = aiEngine.aiEngine(
        apiKey=os.environ["AZURE_OPENAI_API_KEY"],
        baseURL=azure_base,
        apiVersion=api_version,
        primaryModel=chat_deployment,
    )

    store = graphStore.GraphStore()
    db_path = project_root / "database.pkl"
    if db_path.exists():
        store.loadFromFile(str(db_path))

    knowledge_brain = brain.Brain(store, engine, 3)

    mistral_doc_model = (os.environ.get("MISTRAL_OCR_MODEL") or "").strip() or (
        "mistral-document-ai-2512"
    )
    mistral_ep = _normalize_mistral_ocr_endpoint(os.environ["MISTRAL_OCR_ENDPOINT"])
    file_encoder = Encoder(
        mistral_key,
        mistral_ep,
        model=mistral_doc_model,
    )

    s3 = boto3.client("s3")
    bucket = os.environ["S3_BUCKET"]

    entries = sorted(os.listdir(notes_dir))
    _progress(f"Processing {len(entries)} entries from {notes_dir}")

    seen_sources = _indexed_source_paths(store)

    for entry in entries:
        local_path = notes_dir / entry
        if not local_path.is_file():
            continue

        source_key = str(local_path.expanduser().resolve(strict=False))
        if source_key in seen_sources:
            _progress(f"Skipping already indexed: {entry}")
            continue

        _progress(f"Uploading to S3: {entry}")
        s3.upload_file(str(local_path), bucket, entry)

        _progress(f"OCR (Mistral)… this can take minutes for large PDFs: {entry}")
        knowledge_chunks = file_encoder.encodeDocuments(str(local_path))
        _progress(f"OCR produced {len(knowledge_chunks)} page(s)/chunk(s): {entry}")

        for chunk_i, knowledge_chunk in enumerate(knowledge_chunks):
            node_uuid = str(uuid.uuid4())
            chunk_path = chunks_dir / f"{node_uuid}.txt"
            chunk_path.write_text(knowledge_chunk, encoding="utf-8")

            _progress(
                f"Summarizing + linking chunk {chunk_i + 1}/{len(knowledge_chunks)} "
                f"from {entry}…"
            )
            summary = engine.summarize(knowledge_chunk)
            new_node = graphStore.Node(
                node_uuid,
                summary,
                0,
                str(local_path),
            )
            store.addNode(new_node)
            knowledge_brain.addKnowledge(new_node, 3, 0)

        seen_sources.add(source_key)

    _progress("Building graph snapshot…")
    knowledge_brain.print()
    store.exportToFile(str(db_path))
    _progress("Done. Wrote database to database.pkl")


if __name__ == "__main__":
    main()
