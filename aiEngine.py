import os

import json

from openai import APIStatusError, AzureOpenAI
from pydantic import BaseModel, Field


class _ClusterTopic(BaseModel):
    topic: str = Field(
        description="At most three words naming the shared theme of the excerpts."
    )


class _EdgeLabelRow(BaseModel):
    uuid1: str = Field(description="Must match the corresponding input uuid1 exactly.")
    uuid2: str = Field(description="Must match the corresponding input uuid2 exactly.")
    relationship: str = Field(
        description="Short free-form phrase describing how the two topics connect."
    )


class _EdgeLabelsOut(BaseModel):
    edges: list[_EdgeLabelRow] = Field(
        description="Exactly one entry per input topic pair; UUIDs must match inputs."
    )


class aiEngine:
    def __init__(
        self,
        apiKey,
        baseURL,
        apiVersion,
        primaryModel,
        embeddingDeployment: str | None = None,
    ):
        self.client = AzureOpenAI(
            api_key=apiKey,
            api_version=apiVersion,
            azure_endpoint=baseURL,
        )

        self.primaryModel = primaryModel
        self.embeddingDeployment = (embeddingDeployment or "").strip() or None

    def _embedding_deployment(self) -> str:
        if self.embeddingDeployment:
            return self.embeddingDeployment
        for key in (
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
        ):
            raw = (os.environ.get(key) or "").strip()
            if raw:
                return raw
        return "text-embedding-3-large"

    def _embeddings_create(self, *, model: str, input_data):
        """Embeddings call with Azure-specific 404 hints."""
        try:
            return self.client.embeddings.create(model=model, input=input_data)
        except APIStatusError as exc:
            if getattr(exc, "status_code", None) != 404:
                raise
            raise SystemExit(
                "Azure OpenAI embeddings returned HTTP 404 (DeploymentNotFound). "
                "Chat and embeddings use **different deployments** on Azure: create an embeddings "
                "model in your resource (Azure AI Studio → Deployments → Deploy model → choose an "
                "embedding SKU such as text-embedding-3-small or text-embedding-ada-002), then set "
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT in .env to the deployment **name** exactly as "
                "shown in the Deployments list.\n"
                f"Currently calling embeddings with deployment name: {model!r}.\n"
                "The old summarize/link pipeline never called embeddings; clustering requires this deployment."
            ) from exc

    def embed(self, text: str) -> list[float]:
        model = self._embedding_deployment()
        result = self._embeddings_create(model=model, input_data=text)
        return result.data[0].embedding

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        model = self._embedding_deployment()
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self._embeddings_create(model=model, input_data=batch)
            ordered = sorted(result.data, key=lambda d: d.index)
            out.extend(r.embedding for r in ordered)
        return out

    def cluster_topic_label(self, excerpt_texts: list[str]) -> str:
        trimmed = [t.strip() for t in excerpt_texts if t.strip()]
        joined = "\n---\n".join(trimmed[:8])
        system_prompt = (
            "You assign short topic titles to groups of related text excerpts. "
            "The topic must be at most THREE words total "
            "(hyphenated compounds count as one word). No full sentences."
        )
        response = self.client.chat.completions.parse(
            model=self.primaryModel,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": joined or "(empty excerpts)"},
            ],
            response_format=_ClusterTopic,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None or not parsed.topic.strip():
            return "Unlabeled topic"
        words = parsed.topic.strip().split()
        return " ".join(words[:3])

    def edge_labels_for_pairs(
        self,
        pairs_payload: list[dict],
        *,
        contrast_cluster_id: int | None = None,
        contrast_clusters_total: int | None = None,
        contrast_cluster_pair_count: int | None = None,
        sub_batch_pair_count: int | None = None,
    ) -> list[str]:
        """Return relationship strings in the same order as pairs_payload (uuid1 < uuid2).

        When contrast-cluster metadata is set, the prompt explains the embedding-delta
        heuristic so the model can reason about shared contrast geometry (weak signal).
        """
        if not pairs_payload:
            return []

        heuristic = ""
        if (
            contrast_cluster_id is not None
            and contrast_clusters_total is not None
            and contrast_cluster_pair_count is not None
        ):
            sb = sub_batch_pair_count if sub_batch_pair_count is not None else len(pairs_payload)
            heuristic = (
                "Heuristic context (weak signal, not ground truth): Every pair in this request "
                "was placed in the same unsupervised group because the **normalized difference** "
                "between the two topics' centroid embeddings—taken in fixed UUID order "
                "(subtract earlier UUID topic from later UUID topic)—points in a similar direction "
                "in embedding space as the other pairs here. "
                f"This batch is **ContrastGeometryCluster {contrast_cluster_id}** "
                f"(one of **{contrast_clusters_total}** such clusters). "
                f"The full cluster has **{contrast_cluster_pair_count}** pair(s); "
                f"this call lists **{sb}** pair(s). "
                "Treat that shared geometry as a hypothesis that these pairs might involve **analogous kinds** "
                "of semantic contrast, refinement, abstraction shift, or thematic progression. "
                "Use labels and excerpts as primary evidence; use the geometry hint to notice parallel "
                "relationship **patterns** across pairs when the text supports it, and ignore the hint when it clearly conflicts."
            )

        system_prompt = (
            "You describe conceptual relationships between pairs of knowledge topics. "
            "For each input pair, output exactly one row with the same uuid1 and uuid2 strings "
            "and a short free-form relationship phrase (no fixed taxonomy). "
            "Cover every pair once."
        )
        if heuristic:
            system_prompt += "\n\n" + heuristic
        user_prompt = json.dumps(pairs_payload, ensure_ascii=False)
        try:
            response = self.client.chat.completions.parse(
                model=self.primaryModel,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=_EdgeLabelsOut,
            )
        except APIStatusError:
            raise
        parsed = response.choices[0].message.parsed
        mapping: dict[tuple[str, str], str] = {}
        if parsed is not None:
            for row in parsed.edges:
                k = (row.uuid1.strip(), row.uuid2.strip())
                mapping[k] = (row.relationship or "").strip()
        fallback = "Related topics"
        out: list[str] = []
        for p in pairs_payload:
            k = (p["uuid1"], p["uuid2"])
            out.append(mapping.get(k, fallback) or fallback)
        return out

    def summarize(self, text):
        system_prompt = "summarize the following text in 2-3 sentences"
        user_prompt = text

        try:
            response = self.client.chat.completions.create(
                model=self.primaryModel,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except APIStatusError as exc:
            if getattr(exc, "status_code", None) == 404:
                raise SystemExit(
                    "Azure OpenAI returned HTTP 404 (resource not found) for chat completions. "
                    "Typical fixes: ensure AZURE_OPENAI_ENDPOINT looks like "
                    "`https://<your-resource-name>.openai.azure.com` with **no extra path** after the host; "
                    "ensure AZURE_OPENAI_CHAT_DEPLOYMENT equals the **Deployments** "
                    "**name** exactly (not necessarily the SKU model id); if you recently changed API "
                    "preview date, set optional env AZURE_OPENAI_API_VERSION to the value shown in Azure "
                    "portal for your resource."
                ) from exc
            raise

        content = response.choices[0].message.content
        if content is None:
            return ""
        return content
