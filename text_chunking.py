"""Split OCR page markdown into smaller overlapping character windows."""


def split_pages_into_chunks(
    pages: list[str],
    max_chars: int = 900,
    overlap: int = 120,
) -> list[str]:
    """Return flat list of non-empty text segments (per-page sliding windows)."""
    if overlap >= max_chars:
        overlap = max(0, max_chars // 5)

    out: list[str] = []
    for page in pages:
        text = (page or "").strip()
        if not text:
            continue
        start = 0
        n = len(text)
        while start < n:
            end = min(start + max_chars, n)
            piece = text[start:end].strip()
            if piece:
                out.append(piece)
            if end >= n:
                break
            start = end - overlap
    return out
