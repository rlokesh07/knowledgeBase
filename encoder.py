import base64
import json
import mimetypes
from pathlib import Path

import requests

from pdf_split import ocr_max_file_bytes, split_pdf_by_size


class Encoder:
    """Mistral Document AI on Azure (serverless) — follows Microsoft curl samples."""

    def __init__(self, apiKey, endpoint, model="mistral-document-ai-2512"):
        self.apiKey = apiKey
        self.endpoint = endpoint
        self.model = model

    def encodeDocument(self, file):
        path = Path(file)
        max_bytes = ocr_max_file_bytes()
        parts, temp_paths = split_pdf_by_size(path, max_bytes)

        if len(parts) > 1:
            print(
                f"  PDF exceeds {max_bytes / (1024 * 1024):.0f} MB OCR limit "
                f"({path.stat().st_size / (1024 * 1024):.1f} MB) — "
                f"splitting into {len(parts)} part(s): {path.name}",
                flush=True,
            )

        try:
            markdown_files: list[str] = []
            for i, part in enumerate(parts, start=1):
                if len(parts) > 1:
                    print(
                        f"  OCR part {i}/{len(parts)} "
                        f"({part.stat().st_size / (1024 * 1024):.1f} MB)…",
                        flush=True,
                    )
                markdown_files.extend(self._encode_single_document(part))
            return markdown_files
        finally:
            for tmp in temp_paths:
                tmp.unlink(missing_ok=True)

    def _encode_single_document(self, path: Path) -> list[str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.apiKey}",
        }

        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = "application/octet-stream"

        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        data_url = f"data:{mime_type};base64,{encoded}"

        if mime_type.startswith("image/"):
            payload = {
                "model": self.model,
                "document": {"type": "image_url", "image_url": data_url},
                "include_image_base64": True,
            }
        else:
            payload = {
                "model": self.model,
                "document": {"type": "document_url", "document_url": data_url},
                "include_image_base64": True,
            }

        post_url = (self.endpoint or "").strip()
        response = requests.post(post_url, json=payload, headers=headers, timeout=600)

        if response.status_code == 404:
            try:
                blob = response.json()
            except requests.exceptions.JSONDecodeError:
                blob = {}
            err = (blob or {}).get("error") or {}
            if err.get("code") == "DeploymentNotFound":
                raise SystemExit(
                    "Mistral Document AI: Azure returned DeploymentNotFound for "
                    f"model/deployment {self.model!r}. Set MISTRAL_OCR_MODEL in .env to "
                    "the deployment name from the Mistral deployment page "
                    "(e.g. mistral-document-ai-2512)."
                )

        if response.status_code in (401, 403):
            try:
                b = response.json()
                detail_msg = json.dumps(b.get("error") or b)[:800]
            except (ValueError, TypeError, requests.exceptions.JSONDecodeError):
                detail_msg = repr((response.text or "")[:400])
            raise SystemExit(
                "Mistral OCR HTTP "
                f"{response.status_code}: access denied or forbidden. Copy "
                "AZURE_API_KEY (or MISTRAL_API_KEY) from the Mistral Document AI "
                "deployment on Deployments + Endpoint — it must not be only your "
                "Azure OpenAI chat resource key unless that key explicitly covers OCR.\n"
                f"Server detail: {detail_msg}"
            )

        if response.status_code == 400:
            detail = _ocr_error_detail(response)
            if "too large" in detail.lower():
                raise SystemExit(
                    f"Mistral OCR rejected {path.name}: file too large. "
                    f"Limit is ~30 MB; this part is {path.stat().st_size / (1024 * 1024):.1f} MB.\n"
                    f"Server detail: {detail[:600]}"
                )
            raise SystemExit(
                f"Mistral OCR HTTP 400 for {path.name}.\nServer detail: {detail[:800]}"
            )

        response.raise_for_status()

        raw_len = len(response.content or b"")
        if raw_len == 0 and response.status_code == 200:
            raise SystemExit(
                "Mistral Document AI OCR returned HTTP 200 with an empty response body. "
                "Verify MISTRAL_OCR_ENDPOINT is the OCR URL "
                "(…/providers/mistral/azure/ocr) from Deployments + Endpoint, "
                "and that AZURE_API_KEY or MISTRAL_API_KEY belongs to that deployment."
            )

        try:
            body = response.json()
        except requests.exceptions.JSONDecodeError:
            snippet = repr((response.text or "")[:500])
            raise SystemExit(
                f"OCR response was not valid JSON "
                f"(HTTP {response.status_code}, "
                f"content-type={response.headers.get('Content-Type')!r}). "
                f"Preview: {snippet}"
            )

        markdown_files = []
        for page in body.get("pages", []):
            md = page.get("markdown")
            if md is not None:
                markdown_files.append(md)

        return markdown_files

    def encodeDocuments(self, file_path):
        """Same as encodeDocument; name matches callers that process one file at a time."""
        return self.encodeDocument(file_path)


def _ocr_error_detail(response: requests.Response) -> str:
    try:
        blob = response.json()
    except requests.exceptions.JSONDecodeError:
        return (response.text or "")[:800]

    err = blob.get("error") or blob
    msg = err.get("message", "") if isinstance(err, dict) else str(err)
    if isinstance(msg, str) and msg.startswith("{"):
        try:
            inner = json.loads(msg)
            return str(inner.get("message") or inner)
        except json.JSONDecodeError:
            pass
    return str(msg or blob)[:800]
