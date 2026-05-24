"""Assess whether a web source is credible enough to ingest into the graph."""

import json
import os

from pydantic import BaseModel, Field

import aiEngine
from aiEngine import _content_filter_blocked


class SourceAssessment(BaseModel):
    valid: bool = Field(
        description="True if the source is credible and relevant enough to add to the knowledge graph."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the validity judgment (0–1)."
    )
    reasoning: str = Field(
        description="Brief explanation of why the source was accepted or rejected."
    )
    publisher: str = Field(
        description="Identified publisher or organization, or 'unknown'."
    )
    source_type: str = Field(
        description="One of: news, official, academic, industry, blog, aggregator, social, unknown."
    )


class _AssessmentOut(BaseModel):
    assessment: SourceAssessment


def min_validity_confidence() -> float:
    raw = (os.environ.get("AGENT_MIN_SOURCE_CONFIDENCE") or "0.65").strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.65


def assess_source(
    engine: aiEngine.aiEngine,
    *,
    url: str,
    title: str,
    content: str,
    research_goal: str,
) -> SourceAssessment | None:
    """Return a validity assessment, or None if the request was content-filtered."""
    snippet = (content or "").strip()
    if len(snippet) > 6000:
        snippet = snippet[:6000]

    system_prompt = (
        "You assess whether a web source should be added to a knowledge graph. "
        "Accept sources that are credible, attributable, and relevant to the research goal. "
        "Prefer established news outlets, official institutions, academic publishers, "
        "and primary documents. "
        "Reject spam, SEO farms, unattributed reposts, obvious misinformation, "
        "paywalled stubs with no usable content, and sources irrelevant to the goal. "
        "Be skeptical but not overly restrictive for major news organizations. "
        "Set valid=true only when you would trust this source as evidence for the research goal."
    )
    user_prompt = json.dumps(
        {
            "research_goal": research_goal,
            "url": url,
            "title": title,
            "content_excerpt": snippet,
        },
        ensure_ascii=False,
    )

    try:
        response = engine.client.chat.completions.parse(
            model=engine.primaryModel,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=_AssessmentOut,
        )
    except Exception as exc:
        if _content_filter_blocked(exc):
            return None
        raise

    parsed = response.choices[0].message.parsed
    if parsed is None:
        return None
    return parsed.assessment


def is_acceptable(assessment: SourceAssessment | None) -> bool:
    if assessment is None:
        return False
    if not assessment.valid:
        return False
    return assessment.confidence >= min_validity_confidence()
