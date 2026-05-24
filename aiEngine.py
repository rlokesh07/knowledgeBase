import os

import json

from openai import APIStatusError, AzureOpenAI, ContentFilterFinishReasonError
from pydantic import BaseModel, Field


def _content_filter_blocked(exc: BaseException) -> bool:
    """True when Azure/OpenAI rejected or emptied output due to content policy."""
    if isinstance(exc, ContentFilterFinishReasonError):
        return True

    body = getattr(exc, "body", None) if isinstance(exc, APIStatusError) else None
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            body = None

    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            if err.get("code") == "content_filter":
                return True
            inner = err.get("innererror")
            if isinstance(inner, dict) and inner.get("code") == "ResponsibleAIPolicyViolation":
                return True

    msg = str(exc).lower()
    return (
        "content_filter" in msg
        or "content management policy" in msg
        or "responsibleaipolicyviolation" in msg
    )


_RELATIONSHIP_TYPES_PROMPT = """
RELATIONSHIP TYPES — use exactly these strings for relationship_type:

INFLUENCES: Directional. Object A's state changes causally affect Object B's state, but A has no authority or structural power over B. Use as the fallback for causal connections that don't fit a more specific type.

DEPENDS_ON: Directional. Object A cannot maintain its current state or function without Object B. Removing or destabilizing B would directly threaten A's ability to operate normally.

CONTROLS: Directional. Object A has authority, governance, leverage, or decision-making power over Object B's state. A can intentionally change B's state (formal or informal control).

REPRESENTS: Directional. Object A acts on behalf of, speaks for, or embodies Object B. A's decisions and actions are treated as the decisions and actions of B.

CONTAINS: Directional. Object A structurally contains, encompasses, or is the parent body of Object B (geographic, organizational, or part-whole nesting).

Only emit a relationship if the chunk text provides clear evidence. If CONTROLS applies, do not also emit INFLUENCES for the same direction. Emit nothing for pairs with no supported relationship.
""".strip()


class _RelationshipFound(BaseModel):
    source_uuid: str = Field(description="UUID of the source object (must match an input uuid_a or uuid_b exactly).")
    target_uuid: str = Field(description="UUID of the target object (must match an input uuid_a or uuid_b exactly).")
    relationship_type: str = Field(
        description="One of: INFLUENCES, DEPENDS_ON, CONTROLS, REPRESENTS, CONTAINS."
    )
    mechanism: str = Field(
        description=(
            "A detailed explanation of how and why the source object's state change affects the target object. "
            "Must answer three questions: (1) How does the source affect the target — the causal pathway? "
            "(2) Why does that pathway exist — the underlying reason? "
            "(3) Under what conditions does it activate or strengthen, if applicable? "
            "Draw primarily from the text passage, supplemented by logical inference where the text is incomplete. "
            "Capture all relevant causal factors mentioned, not just the primary one. "
            "Write 2–4 sentences. Do not simply restate the relationship type."
        )
    )


class _RelationshipsOut(BaseModel):
    relationships: list[_RelationshipFound] = Field(
        description="All directed typed relationships found across all input pairs. Empty list if none."
    )


class _ObjectEntry(BaseModel):
    name: str = Field(
        description=(
            "Canonical short name for the object "
            "(e.g. 'Amazon River', 'photosynthesis', 'Marie Curie')."
        )
    )
    object_type: str = Field(
        description=(
            "One of: person, place, organization, concept, event, "
            "artifact, species, process, or other."
        )
    )
    description: str = Field(
        description="One sentence describing this object's role or nature in context."
    )
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "1–3 lowercase topical theme tags for grouping and filtering "
            "(e.g. '2028 united states president', 'amazon rainforest', 'gulf remittances')."
        ),
    )


class _ExtractedObjects(BaseModel):
    objects: list[_ObjectEntry] = Field(
        description=(
            "All distinct named objects found in the excerpt. "
            "Omit generic nouns, vague references, and pronouns."
        )
    )


class _ObjectStateRow(BaseModel):
    uuid: str = Field(description="Must match an input uuid exactly.")
    state: str = Field(
        description=(
            "A comprehensive, free-form snapshot of the object's current condition across "
            "all relevant dimensions the excerpts support — political, military, economic, "
            "diplomatic, geographic, social, or any other applicable dimension. "
            "Not a single label; a dense, information-rich portrait at a specific point in time. "
            "Extract and consolidate all relevant details from the excerpts, supplementing with "
            "logical inference only where the text leaves gaps. Write 3–6 sentences."
        )
    )


class _ObjectStatesOut(BaseModel):
    states: list[_ObjectStateRow] = Field(
        description="Exactly one state entry per input object that has usable excerpt evidence."
    )


class _ResearchPlan(BaseModel):
    needs_research: bool = Field(
        description="True if external web research is needed beyond the provided graph context."
    )
    search_queries: list[str] = Field(
        description="Up to 3 focused web search queries if needs_research is true; else empty."
    )
    reasoning: str = Field(description="Why research is or is not needed.")


class _InstrumentSuggestion(BaseModel):
    id: str = Field(
        description="Stable slug id, e.g. brent_crude or try_usd."
    )
    symbol: str = Field(
        description="Yahoo Finance ticker symbol, e.g. BZ=F, DX-Y.NYB, ^GSPC."
    )
    name: str = Field(description="Human-readable instrument label.")
    asset_class: str = Field(
        description="One of: fx, commodity, equity, bond, index, crypto, other."
    )
    reasoning: str = Field(
        description="One sentence on why this instrument matters for the question."
    )


class _InstrumentPlan(BaseModel):
    needs_instruments: bool = Field(
        description="True if live financial instrument prices/levels would materially help answer the question."
    )
    instruments: list[_InstrumentSuggestion] = Field(
        description="Up to AGENT_INSTRUMENTS_MAX new instruments if needs_instruments; else empty."
    )
    reasoning: str = Field(description="Why live instrument data is or is not needed.")


class _SeedChangeOut(BaseModel):
    seed_uuid: str = Field(description="UUID of the seed node (must match a candidate uuid).")
    hypothetical_change: str = Field(
        description="Short summary of the scenario change to propagate to neighbors."
    )
    new_state: str = Field(description="Full revised state text for the seed node under this scenario.")
    reasoning: str = Field(description="Why this node and scenario answer the question.")


class _PropagationEvalOut(BaseModel):
    significant: bool = Field(
        description="True if the incoming change materially warrants updating this node's state."
    )
    new_state: str = Field(
        default="",
        description="Full revised state if significant; empty if not significant.",
    )
    change_to_propagate: str = Field(
        default="",
        description="Short summary of what changed on this node for downstream propagation.",
    )
    reasoning: str = Field(default="")


class _PropagationReportOut(BaseModel):
    answer: str = Field(
        description="User-facing report summarizing the scenario, cascade of significant changes, "
        "and any market or instrument shifts."
    )


class _AgentAnswer(BaseModel):
    answer: str = Field(description="Complete answer to the user's question.")
    sources_used: list[str] = Field(
        description="URLs or graph sources cited in the answer."
    )


class _MarketRelevanceRow(BaseModel):
    market_id: str = Field(description="Must match input market id exactly.")
    relevant: bool = Field(
        description="True if this prediction market materially relates to entities or themes in the ontology."
    )
    reasoning: str = Field(description="One sentence explaining the relevance decision.")


class _MarketRelevanceOut(BaseModel):
    results: list[_MarketRelevanceRow] = Field(
        description="One row per input market."
    )


class _NodeTagRow(BaseModel):
    uuid: str = Field(description="Must match an input uuid exactly.")
    tags: list[str] = Field(
        description=(
            "1–3 lowercase topical theme tags describing what this node is about "
            "(e.g. '2028 united states president', 'us iran relations', 'polymarket')."
        )
    )


class _NodeTagsOut(BaseModel):
    rows: list[_NodeTagRow] = Field(
        description="Exactly one row per input uuid."
    )


class _MarketLinkRow(BaseModel):
    market_uuid: str = Field(description="UUID of the market node (must match input).")
    target_uuid: str = Field(description="UUID of the ontology object materially affected if the market resolves Yes.")
    mechanism: str = Field(
        description=(
            "If the market's affirmative outcome occurs, explain the real-world significance for the "
            "target object — how it would materially change that object's political, military, economic, "
            "diplomatic, geographic, or operational condition. "
            "Do NOT mention Polymarket, odds, probabilities, traders, or market mechanics. "
            "Write 2–4 sentences about downstream consequences for the target object only."
        )
    )


class _MarketLinksOut(BaseModel):
    links: list[_MarketLinkRow] = Field(
        description=(
            "Only market→object links where the market's resolution would materially affect the target "
            "object's real-world state. Omit tangential name mentions with no genuine significance."
        )
    )


class _InstrumentLinkRow(BaseModel):
    instrument_uuid: str = Field(description="UUID of the instrument node (must match input).")
    target_uuid: str = Field(description="UUID of the ontology object materially affected by this instrument.")
    relationship_type: str = Field(
        description="Exactly one of: INFLUENCES, DEPENDS_ON."
    )
    mechanism: str = Field(
        description=(
            "Explain the real-world economic or financial significance: how movements in this instrument's "
            "price or level materially affect the target object's political, military, economic, diplomatic, "
            "or operational condition. "
            "Do NOT describe ticker mechanics, chart patterns, or trader behavior. "
            "Write 2–4 sentences about downstream consequences for the target object only."
        )
    )


class _InstrumentLinksOut(BaseModel):
    links: list[_InstrumentLinkRow] = Field(
        description=(
            "Only instrument→object links where price/level movements materially affect the target object. "
            "Omit tangential domain overlap without genuine economic significance."
        )
    )


class _MarketMarketLinkRow(BaseModel):
    source_uuid: str = Field(description="UUID of the source market (must match input).")
    target_uuid: str = Field(description="UUID of the target market (must match catalog).")
    relationship_type: str = Field(
        description="Exactly one of: INFLUENCES, DEPENDS_ON."
    )
    mechanism: str = Field(
        description=(
            "Explain the logical implication: if the source market resolves Yes, how does that constrain "
            "or change whether/how the target market can resolve Yes? "
            "Do NOT mention Polymarket odds, probabilities, or traders. "
            "Write 2–4 sentences about the logical dependency between the two outcomes."
        )
    )


class _MarketMarketLinksOut(BaseModel):
    links: list[_MarketMarketLinkRow] = Field(
        description=(
            "Only market→market links where the source resolving Yes logically implies or constrains "
            "the target. Use DEPENDS_ON when target Yes requires source Yes; INFLUENCES when source Yes "
            "materially affects but does not strictly require target Yes. Omit topical similarity without implication."
        )
    )


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

    def extract_relationships(
        self, chunk_text: str, node_pairs_payload: list[dict]
    ) -> list[_RelationshipFound]:
        """Return all directed typed relationships found in chunk_text for the given node pairs.

        Each entry in node_pairs_payload must have keys:
            uuid_a, label_a, description_a, uuid_b, label_b, description_b
        """
        if not node_pairs_payload:
            return []

        system_prompt = (
            "You identify directed typed relationships between pairs of named objects. "
            "Use the provided text passage as context, and also apply your world knowledge "
            "and logical reasoning to infer relationships that are plausible given what the "
            "passage discusses — even if not explicitly stated word-for-word. "
            "For each pair, output every (source_uuid, target_uuid, relationship_type, mechanism) entry "
            "that is reasonably supported or inferable. "
            "Only omit a relationship if it is clearly contradicted by the passage or "
            "there is genuinely no plausible connection between the two objects.\n\n"
            "For each relationship, write a mechanism: a detailed causal explanation of how and why "
            "the source object's state change affects the target object. The mechanism must answer: "
            "(1) How does the source affect the target — the causal pathway? "
            "(2) Why does that pathway exist — the underlying reason? "
            "(3) Under what conditions does it activate or strengthen, if applicable? "
            "Draw primarily from the text passage and supplement with logical inference where the text is incomplete. "
            "Capture all relevant causal factors, not just the primary one. Write 2–4 sentences.\n\n"
            + _RELATIONSHIP_TYPES_PROMPT
        )
        user_prompt = (
            "TEXT PASSAGE:\n"
            + chunk_text[:3000]
            + "\n\nOBJECT PAIRS:\n"
            + json.dumps(node_pairs_payload, ensure_ascii=False)
        )
        try:
            response = self.client.chat.completions.parse(
                model=self.primaryModel,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=_RelationshipsOut,
            )
        except (ContentFilterFinishReasonError, APIStatusError) as exc:
            if _content_filter_blocked(exc):
                return []
            raise
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return []
        valid_uuids = {p["uuid_a"] for p in node_pairs_payload} | {p["uuid_b"] for p in node_pairs_payload}
        valid_types = {"INFLUENCES", "DEPENDS_ON", "CONTROLS", "REPRESENTS", "CONTAINS"}
        return [
            r for r in parsed.relationships
            if r.source_uuid in valid_uuids
            and r.target_uuid in valid_uuids
            and r.source_uuid != r.target_uuid
            and r.relationship_type in valid_types
        ]

    def cluster_topic_label(self, excerpt_texts: list[str]) -> str:
        trimmed = [t.strip() for t in excerpt_texts if t.strip()]
        joined = "\n---\n".join(trimmed[:8])
        system_prompt = (
            "You assign short topic titles to groups of related text excerpts. "
            "The topic must be at most THREE words total "
            "(hyphenated compounds count as one word). No full sentences."
        )
        try:
            response = self.client.chat.completions.parse(
                model=self.primaryModel,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": joined or "(empty excerpts)"},
                ],
                response_format=_ClusterTopic,
            )
        except (ContentFilterFinishReasonError, APIStatusError) as exc:
            if _content_filter_blocked(exc):
                return "Unlabeled topic"
            raise
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
        except (ContentFilterFinishReasonError, APIStatusError) as exc:
            if _content_filter_blocked(exc):
                return ["Related topics"] * len(pairs_payload)
            raise
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

    def extract_objects(self, chunk_texts: list[str]) -> list[_ObjectEntry]:
        """Extract distinct named objects from one or more text chunks bundled together."""
        chunk_texts = [t for t in chunk_texts if (t or "").strip()]
        if not chunk_texts:
            return []
        combined = "\n\n---\n\n".join(chunk_texts)
        system_prompt = (
            "You extract distinct named objects from text. "
            "An object is any distinct entity — physical, living, geographic, or conceptual — "
            "that exists within an ecosystem and can be connected to other objects through relationships. "
            "Each object has properties describing its current state and influences other objects. "
            "Extract only clearly named, specific objects. "
            "Omit vague references, pronouns, generic nouns (e.g. 'the study', 'a method'), "
            "and numerical values. "
            "For each object provide: name (canonical short form), object_type, "
            "a one-sentence description, and 1–3 lowercase topical tags for grouping "
            "(e.g. '2028 united states president', 'middle east conflict')."
        )
        try:
            response = self.client.chat.completions.parse(
                model=self.primaryModel,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": combined},
                ],
                response_format=_ExtractedObjects,
            )
        except (ContentFilterFinishReasonError, APIStatusError) as exc:
            if _content_filter_blocked(exc):
                return []
            raise
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return []
        return [o for o in parsed.objects if (o.name or "").strip()]

    def extract_object_states(
        self, objects_payload: list[dict]
    ) -> tuple[list[_ObjectStateRow], bool]:
        """Return current-state snapshots for each object, grounded in its excerpt texts.

        Each entry in objects_payload must have keys:
            uuid, label, object_type, description, excerpts
        """
        if not objects_payload:
            return [], False

        system_prompt = (
            "You write current-state snapshots for named objects in a knowledge ecosystem. "
            "For each object, produce a comprehensive free-form description of its current condition "
            "across every relevant dimension the excerpts support — political, military, economic, "
            "diplomatic, geographic, social, or any other dimension the source material speaks to. "
            "The state is NOT a single label or category; it is a dense, information-rich portrait "
            "of the object at a specific point in time. "
            "Extract and consolidate all relevant details from the excerpts into one holistic snapshot. "
            "Supplement with logical inference only where the excerpts leave gaps. "
            "Pack in as much supported context as possible so downstream reasoning about causal "
            "propagation is well-grounded. Write 3–6 sentences per object. "
            "Return exactly one row per input uuid that has usable evidence; omit objects with "
            "no meaningful information in their excerpts."
        )
        user_prompt = json.dumps(objects_payload, ensure_ascii=False)
        try:
            response = self.client.chat.completions.parse(
                model=self.primaryModel,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=_ObjectStatesOut,
            )
        except Exception as exc:
            if _content_filter_blocked(exc):
                return [], True
            raise
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return [], False
        valid_uuids = {p["uuid"] for p in objects_payload}
        rows = [r for r in parsed.states if r.uuid in valid_uuids and (r.state or "").strip()]
        return rows, False

    def plan_research(self, question: str, graph_context: str) -> _ResearchPlan | None:
        system_prompt = (
            "You decide whether a knowledge-graph agent needs fresh web research to answer a question. "
            "Use the provided graph context first. "
            "Set needs_research=true only when the context is insufficient, stale for a time-sensitive question, "
            "or missing key facts. "
            "If research is needed, propose up to 3 precise search queries (not full sentences as questions)."
        )
        user_prompt = (
            f"QUESTION:\n{question}\n\nGRAPH CONTEXT:\n{graph_context[:12000]}"
        )
        try:
            response = self.client.chat.completions.parse(
                model=self.primaryModel,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=_ResearchPlan,
            )
        except Exception as exc:
            if _content_filter_blocked(exc):
                return None
            raise
        return response.choices[0].message.parsed

    def plan_instruments(
        self,
        question: str,
        graph_context: str,
        existing_instruments: list[dict],
        *,
        max_instruments: int = 5,
    ) -> _InstrumentPlan | None:
        system_prompt = (
            "You decide whether a knowledge-graph agent needs live financial instrument data "
            "to answer a question well. "
            "Set needs_instruments=true when the question involves economic sensitivity, currency moves, "
            "commodity prices, equity/rates context, fiscal or trade exposure, or similar market-linked analysis. "
            "If live prices would not materially improve the answer, set needs_instruments=false. "
            "When proposing instruments, use concrete Yahoo Finance symbols (not vague names). "
            "Examples: Brent crude BZ=F, WTI CL=F, US Dollar Index DX-Y.NYB, gold GC=F, S&P 500 ^GSPC, "
            "10-year yield ^TNX, EUR/USD EURUSD=X. "
            "Do NOT re-propose instruments already in existing_instruments (match by symbol or id). "
            f"Propose at most {max_instruments} new instruments."
        )
        user_prompt = json.dumps(
            {
                "question": question,
                "graph_context": graph_context[:12000],
                "existing_instruments": existing_instruments,
            },
            ensure_ascii=False,
        )
        try:
            response = self.client.chat.completions.parse(
                model=self.primaryModel,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=_InstrumentPlan,
            )
        except Exception as exc:
            if _content_filter_blocked(exc):
                return None
            raise
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return None
        existing_ids = {str(x.get("id") or "").lower() for x in existing_instruments}
        existing_symbols = {
            str(x.get("symbol") or "").upper() for x in existing_instruments
        }
        filtered: list[_InstrumentSuggestion] = []
        for row in parsed.instruments[:max_instruments]:
            if row.id.lower() in existing_ids or row.symbol.upper() in existing_symbols:
                continue
            if not (row.symbol or "").strip():
                continue
            filtered.append(row)
        parsed.instruments = filtered
        if not filtered:
            parsed.needs_instruments = False
        return parsed

    def select_seed_change(
        self,
        question: str,
        candidates: list[dict],
    ) -> _SeedChangeOut | None:
        """Pick a seed node and hypothetical state change for predict-mode propagation."""
        if not candidates:
            return None

        system_prompt = (
            "You run predict mode on a knowledge graph. Given a user question and candidate nodes, "
            "choose ONE seed node whose state would change under the scenario implied by the question. "
            "Write a hypothetical_change (short, for propagation) and new_state (full revised state text). "
            "The scenario may be counterfactual ('what if X happens?') or analytical ('how would X affect Y?'). "
            "Ground the change in the node's current state and description. "
            "seed_uuid must exactly match a candidate uuid."
        )
        user_prompt = json.dumps(
            {"question": question, "candidates": candidates},
            ensure_ascii=False,
        )
        try:
            response = self.client.chat.completions.parse(
                model=self.primaryModel,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=_SeedChangeOut,
            )
        except Exception as exc:
            if _content_filter_blocked(exc):
                return None
            raise
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return None
        valid = {c["uuid"] for c in candidates}
        if parsed.seed_uuid not in valid or not (parsed.new_state or "").strip():
            return None
        return parsed

    def evaluate_propagated_change(
        self,
        node: dict,
        *,
        incoming_change: str,
        from_label: str,
        edge_mechanism: str,
    ) -> _PropagationEvalOut | None:
        """Micro-agent: decide if an incoming change significantly updates one node."""
        otype = (node.get("object_type") or "").lower()
        live_hint = ""
        if otype == "market":
            live_hint = (
                " This is a Polymarket prediction market. If significant, new_state must describe "
                "revised odds/outlook in plain language (e.g. Yes probability rising to ~60%). "
                "change_to_propagate should note the market shift."
            )
        elif otype == "instrument":
            live_hint = (
                " This is a live financial instrument. If significant, new_state must describe "
                "revised price/level outlook. change_to_propagate should note the market move."
            )

        system_prompt = (
            "You are a short-lived propagation agent. You see ONLY one node and an incoming change "
            "from a connected node in the graph. Decide whether the incoming change materially "
            "warrants updating this node's state. "
            "Set significant=true only when the effect is concrete and non-trivial for this entity. "
            "If significant, write new_state (full revised state) and change_to_propagate "
            "(short summary for further propagation). "
            "If not significant, set significant=false and leave new_state empty. "
            "Do not speculate beyond what the incoming change and edge mechanism support."
            + live_hint
        )
        user_prompt = json.dumps(
            {
                "node": node,
                "incoming_change": incoming_change,
                "from_label": from_label,
                "edge_mechanism": edge_mechanism,
            },
            ensure_ascii=False,
        )
        try:
            response = self.client.chat.completions.parse(
                model=self.primaryModel,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=_PropagationEvalOut,
            )
        except Exception as exc:
            if _content_filter_blocked(exc):
                return None
            raise
        return response.choices[0].message.parsed

    def synthesize_propagation_report(
        self,
        question: str,
        seed: dict,
        changes: list[dict],
    ) -> _PropagationReportOut | None:
        """Summarize the propagation cascade for the user."""
        system_prompt = (
            "You write a clear report of a knowledge-graph predict-mode simulation. "
            "Explain the scenario, the seed change, and each significant downstream effect in order. "
            "Highlight market (prediction odds) and instrument (price/level) shifts in a dedicated section "
            "if any occurred. Be concise but complete. Do not mention internal agent mechanics."
        )
        user_prompt = json.dumps(
            {"question": question, "seed": seed, "changes": changes},
            ensure_ascii=False,
        )
        try:
            response = self.client.chat.completions.parse(
                model=self.primaryModel,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=_PropagationReportOut,
            )
        except Exception as exc:
            if _content_filter_blocked(exc):
                return None
            raise
        return response.choices[0].message.parsed

    def synthesize_answer(
        self,
        question: str,
        graph_context: str,
        web_sources: list[dict],
    ) -> _AgentAnswer | None:
        system_prompt = (
            "You answer questions using a knowledge graph and newly validated web sources. "
            "Prefer graph context; use web sources to fill gaps or update time-sensitive facts. "
            "Cite which sources support key claims. Be precise and concise."
        )
        user_prompt = json.dumps(
            {
                "question": question,
                "graph_context": graph_context[:12000],
                "web_sources": web_sources[:8],
            },
            ensure_ascii=False,
        )
        try:
            response = self.client.chat.completions.parse(
                model=self.primaryModel,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=_AgentAnswer,
            )
        except Exception as exc:
            if _content_filter_blocked(exc):
                return None
            raise
        return response.choices[0].message.parsed

    def assess_market_relevance(
        self,
        ontology_summary: str,
        markets_payload: list[dict],
    ) -> list[_MarketRelevanceRow]:
        """Return relevance judgments for Polymarket markets against the ontology."""
        if not markets_payload:
            return []

        system_prompt = (
            "You judge whether Polymarket prediction markets are relevant to a knowledge-graph ontology. "
            "A market is relevant if its question/resolution criteria connect to named entities, places, "
            "organizations, events, or themes represented in the ontology — even indirectly "
            "(e.g. a market on 'US-Iran peace deal' is relevant when Iran and United States are in the ontology). "
            "Reject markets about unrelated sports, entertainment, crypto prices, or topics with no connection "
            "to the ontology entities/themes. "
            "Return one row per market id."
        )
        user_prompt = json.dumps(
            {
                "ontology_summary": ontology_summary[:8000],
                "markets": markets_payload,
            },
            ensure_ascii=False,
        )
        try:
            response = self.client.chat.completions.parse(
                model=self.primaryModel,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=_MarketRelevanceOut,
            )
        except Exception as exc:
            if _content_filter_blocked(exc):
                return []
            raise
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return []
        valid_ids = {m["id"] for m in markets_payload}
        return [r for r in parsed.results if r.market_id in valid_ids]

    def assign_node_tags(self, nodes_payload: list[dict]) -> list[_NodeTagRow]:
        """Assign topical grouping tags to nodes (objects, markets, etc.)."""
        if not nodes_payload:
            return []

        system_prompt = (
            "You assign topical theme tags to knowledge-graph nodes for filtering and grouping. "
            "Each tag is a short lowercase phrase describing a theme, event, election, conflict, "
            "region, or topic the node is about (e.g. '2028 united states president', "
            "'us iran relations', 'gaza conflict', 'polymarket'). "
            "Use 1–3 tags per node. Tags must be specific enough to filter a coherent subset. "
            "For prediction markets, include a tag for the underlying event or election "
            "(e.g. a market on the 2028 US presidential election → '2028 united states president'). "
            "Return exactly one row per input uuid."
        )
        user_prompt = json.dumps(nodes_payload, ensure_ascii=False)
        try:
            response = self.client.chat.completions.parse(
                model=self.primaryModel,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=_NodeTagsOut,
            )
        except Exception as exc:
            if _content_filter_blocked(exc):
                return []
            raise
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return []
        valid_uuids = {p["uuid"] for p in nodes_payload}
        return [r for r in parsed.rows if r.uuid in valid_uuids]

    def link_markets_to_objects(
        self,
        markets_payload: list[dict],
        objects_payload: list[dict],
    ) -> list[_MarketLinkRow]:
        """Return PREDICTS edges from market nodes to referenced ontology objects."""
        if not markets_payload or not objects_payload:
            return []

        system_prompt = (
            "You link Polymarket prediction markets to objects in a knowledge-graph ontology. "
            "Create a link ONLY when the market's resolution (especially a Yes outcome) would "
            "materially and specifically affect the target object's real-world state — its political "
            "position, military posture, economic condition, diplomatic standing, territorial control, "
            "operational capacity, or similar concrete dimensions. "
            "Do NOT link merely because an object is named in the market text, shares a domain, "
            "or would be indirectly or trivially affected. If there is no real significance, omit the link. "
            "Each link is PREDICTS (market → object): the market tracks whether a consequential event "
            "for that object will occur. "
            "The mechanism must describe the SIGNIFICANCE of the event happening for the target object — "
            "what would change for that object if the market resolves affirmatively. "
            "Never describe Polymarket odds, probabilities, trader sentiment, or market mechanics in the mechanism. "
            "Use the object's current state from the input when explaining how the event would alter it. "
            "Return an empty links list for markets with no objects that meet this significance bar."
        )
        user_prompt = json.dumps(
            {"markets": markets_payload, "objects": objects_payload},
            ensure_ascii=False,
        )
        try:
            response = self.client.chat.completions.parse(
                model=self.primaryModel,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=_MarketLinksOut,
            )
        except Exception as exc:
            if _content_filter_blocked(exc):
                return []
            raise
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return []
        market_ids = {m["uuid"] for m in markets_payload}
        object_ids = {o["uuid"] for o in objects_payload}
        return [
            link
            for link in parsed.links
            if link.market_uuid in market_ids
            and link.target_uuid in object_ids
            and link.market_uuid != link.target_uuid
            and (link.mechanism or "").strip()
        ]

    def link_instruments_to_objects(
        self,
        instruments_payload: list[dict],
        objects_payload: list[dict],
    ) -> list[_InstrumentLinkRow]:
        """Return INFLUENCES/DEPENDS_ON edges from instrument nodes to ontology objects."""
        if not instruments_payload or not objects_payload:
            return []

        valid_types = {"INFLUENCES", "DEPENDS_ON"}
        system_prompt = (
            "You link live financial instruments to objects in a knowledge-graph ontology. "
            "Create a link ONLY when movements in the instrument's price, yield, or index level would "
            "materially and specifically affect the target object's real-world state — its economic condition, "
            "fiscal position, trade balance, currency exposure, energy revenue, import costs, market access, "
            "political stability under economic stress, or similar concrete dimensions. "
            "Do NOT link merely because an object operates in a related sector or is named in passing. "
            "Use exactly these relationship types:\n"
            "DEPENDS_ON — the target's condition is structurally tied to this instrument's level "
            "(e.g. major oil exporter and crude price).\n"
            "INFLUENCES — instrument moves materially shape but do not strictly define the target's condition.\n"
            "The mechanism must describe real-world economic significance for the target — never ticker jargon, "
            "chart patterns, or trader sentiment. Use the instrument's current state and the object's current state "
            "from the input. Return empty links if none qualify."
        )
        user_prompt = json.dumps(
            {"instruments": instruments_payload, "objects": objects_payload},
            ensure_ascii=False,
        )
        try:
            response = self.client.chat.completions.parse(
                model=self.primaryModel,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=_InstrumentLinksOut,
            )
        except Exception as exc:
            if _content_filter_blocked(exc):
                return []
            raise
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return []
        instrument_ids = {m["uuid"] for m in instruments_payload}
        object_ids = {o["uuid"] for o in objects_payload}
        out: list[_InstrumentLinkRow] = []
        for link in parsed.links:
            rel = (link.relationship_type or "").strip().upper()
            if rel not in valid_types:
                rel = "INFLUENCES"
            if (
                link.instrument_uuid in instrument_ids
                and link.target_uuid in object_ids
                and link.instrument_uuid != link.target_uuid
                and (link.mechanism or "").strip()
            ):
                link.relationship_type = rel
                out.append(link)
        return out

    def link_markets_to_markets(
        self,
        source_markets_payload: list[dict],
        catalog_payload: list[dict],
    ) -> list[_MarketMarketLinkRow]:
        """Return INFLUENCES/DEPENDS_ON edges where source market Yes implies/constrains target."""
        if not source_markets_payload or not catalog_payload:
            return []

        valid_types = {"INFLUENCES", "DEPENDS_ON"}
        system_prompt = (
            "You identify logical implication links between Polymarket prediction markets. "
            "Create a directed edge ONLY when resolving the source market Yes logically implies, "
            "requires, or materially constrains whether/how the target market can resolve Yes. "
            "Do NOT link markets that merely share a topic, event family, or domain without a logical "
            "implication between their resolution criteria. "
            "Use exactly these relationship types:\n"
            "DEPENDS_ON — target Yes cannot occur unless source Yes occurs (strict logical prerequisite).\n"
            "INFLUENCES — source Yes materially changes whether target Yes is plausible or meaningful, "
            "but target Yes could still occur without source Yes.\n"
            "The mechanism must explain the logical implication between the two outcomes — what resolving "
            "source Yes means for target Yes — not odds, probabilities, or trader behavior. "
            "Source and target must be different markets. Return empty links if none qualify."
        )
        user_prompt = json.dumps(
            {
                "source_markets": source_markets_payload,
                "target_catalog": catalog_payload,
            },
            ensure_ascii=False,
        )
        try:
            response = self.client.chat.completions.parse(
                model=self.primaryModel,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=_MarketMarketLinksOut,
            )
        except Exception as exc:
            if _content_filter_blocked(exc):
                return []
            raise
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return []
        source_ids = {m["uuid"] for m in source_markets_payload}
        catalog_ids = {m["uuid"] for m in catalog_payload}
        out: list[_MarketMarketLinkRow] = []
        for link in parsed.links:
            if link.source_uuid not in source_ids:
                continue
            if link.target_uuid not in catalog_ids:
                continue
            if link.source_uuid == link.target_uuid:
                continue
            if link.relationship_type not in valid_types:
                continue
            if not (link.mechanism or "").strip():
                continue
            out.append(link)
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
