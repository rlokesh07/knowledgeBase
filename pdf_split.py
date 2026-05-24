"""Split oversized PDFs into OCR-safe parts (Mistral Azure limit is 30 MB)."""

import os
import tempfile
from pathlib import Path

import fitz
from pypdf import PdfReader, PdfWriter

_DEFAULT_MAX_BYTES = 28 * 1024 * 1024  # safety margin under the 30 MB API cap
_RASTER_DPI_STEPS = (200, 150, 120, 100, 72, 50)


def ocr_max_file_bytes() -> int:
    raw = (os.environ.get("MISTRAL_OCR_MAX_FILE_BYTES") or "").strip()
    if not raw:
        return _DEFAULT_MAX_BYTES
    try:
        return max(1024 * 1024, int(raw))
    except ValueError:
        return _DEFAULT_MAX_BYTES


def _write_temp_pdf(writer: PdfWriter) -> Path:
    fd, name = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    path = Path(name)
    with path.open("wb") as f:
        writer.write(f)
    return path


def _part_size(start: int, end: int, reader: PdfReader) -> int:
    writer = PdfWriter()
    for i in range(start, end):
        writer.add_page(reader.pages[i])
    tmp = _write_temp_pdf(writer)
    try:
        return tmp.stat().st_size
    finally:
        tmp.unlink(missing_ok=True)


def _rasterize_page(src: Path, page_index: int, dpi: int) -> Path:
    doc = fitz.open(src)
    try:
        page = doc[page_index]
        pix = page.get_pixmap(dpi=dpi)
        out = fitz.open()
        try:
            img_page = out.new_page(width=pix.width, height=pix.height)
            img_page.insert_image(img_page.rect, pixmap=pix)
            fd, name = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            path = Path(name)
            out.save(path, garbage=4, deflate=True)
            return path
        finally:
            out.close()
    finally:
        doc.close()


def _compress_oversized_page(src: Path, page_index: int, limit: int) -> Path:
    """Rasterize a single page at decreasing DPI until it fits under limit."""
    for dpi in _RASTER_DPI_STEPS:
        part = _rasterize_page(src, page_index, dpi)
        size = part.stat().st_size
        if size <= limit:
            return part
        part.unlink(missing_ok=True)

    raise SystemExit(
        f"PDF page {page_index + 1} of {src.name} could not be compressed below "
        f"{limit / (1024 * 1024):.0f} MB for Mistral OCR."
    )


def split_pdf_by_size(path: Path, max_bytes: int | None = None) -> tuple[list[Path], list[Path]]:
    """Return (paths_to_ocr, temp_paths).

    If the PDF fits within max_bytes, returns ([path], []).
    Otherwise splits into page-range parts; individual pages over the limit
    are rasterized at lower DPI until they fit.
    """
    path = path.expanduser().resolve()
    limit = max_bytes if max_bytes is not None else ocr_max_file_bytes()

    if path.suffix.lower() != ".pdf":
        return [path], []

    file_size = path.stat().st_size
    if file_size <= limit:
        return [path], []

    reader = PdfReader(str(path))
    total_pages = len(reader.pages)
    if total_pages == 0:
        return [path], []

    parts: list[Path] = []
    temps: list[Path] = []
    start = 0

    while start < total_pages:
        end = start + 1
        best_end = start + 1

        while end <= total_pages:
            size = _part_size(start, end, reader)
            if size <= limit:
                best_end = end
                end += 1
            else:
                break

        if best_end == start + 1 and _part_size(start, start + 1, reader) > limit:
            part_path = _compress_oversized_page(path, start, limit)
            parts.append(part_path)
            temps.append(part_path)
            start += 1
            continue

        writer = PdfWriter()
        for i in range(start, best_end):
            writer.add_page(reader.pages[i])
        part_path = _write_temp_pdf(writer)
        parts.append(part_path)
        temps.append(part_path)
        start = best_end

    return parts, temps
