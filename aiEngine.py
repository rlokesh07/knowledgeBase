from openai import APIStatusError, AzureOpenAI
import json
import math
import uuid

from pydantic import BaseModel, Field

import graphStore


def cosineSimilarity(a, b):
    dot = 0
    normA = 0
    normB = 0
    for i in range(len(a)):
        dot += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]

    denom = math.sqrt(normA) * math.sqrt(normB)
    if denom == 0:
        return 0.0
    return dot / denom


class _CandidateEdge(BaseModel):
    uuid2: str = Field(description="UUID of the existing node from the candidate list best related to the new node.")
    relationship: str = Field(
        description="Short description of the conceptual connection between the two notes."
    )


def _serializeNodes(nodes):
    return [
        {"uuid": n.uuid, "summary": n.summary, "layer": n.layer, "link": n.link}
        for n in nodes
    ]


class aiEngine:
    def __init__(self, apiKey, baseURL, apiVersion, primaryModel):
        self.client = AzureOpenAI(
            api_key=apiKey,
            api_version=apiVersion,
            azure_endpoint=baseURL,
        )

        self.primaryModel = primaryModel

    def newEdge(self, newNode, currentNodes):
        system_prompt = (
            "You are given a JSON list of existing knowledge-base nodes "
            "(each has uuid, summary, layer, link). One new node is being indexed. "
            f"The new node's UUID is {newNode.uuid!r}. uuid1 MUST be exactly that UUID. "
            "Choose uuid2 from the candidate list UUIDs only (different from uuid1). "
            "Return structured output with uuid2 and relationship describing the connection."
        )
        serialized = _serializeNodes(currentNodes)
        user_prompt = (
            f"New node summary:\n{newNode.summary!s}\n\n"
            f"Candidate nodes (JSON):\n{json.dumps(serialized)}\n"
        )

        response = self.client.chat.completions.parse(
            model=self.primaryModel,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=_CandidateEdge,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("Model returned no structured edge payload.")

        allowed = {n.uuid for n in currentNodes if n.uuid != newNode.uuid}
        if not allowed:
            raise RuntimeError(
                "newEdge called with no distinct candidate nodes (need another node in the same layer)."
            )
        uuid2 = parsed.uuid2.strip()
        if uuid2 == newNode.uuid or uuid2 not in allowed:
            uuid2 = next(iter(allowed))

        edge_id = str(uuid.uuid4())
        return graphStore.Edge(
            edge_id,
            newNode.uuid,
            uuid2,
            1.0,
            parsed.relationship,
            0,
        )

    def generateQuestion(self, nodes):

        summaries = ""
        for node in nodes:
            summaries += node.summary
            summaries += " - "

        system_prompt = (
            "given the  following nodes, generate an insightful question to connect them. Return only the question"
        )
        user_prompt = summaries

        response = self.client.chat.completions.create(
            model=self.primaryModel,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def embed(self, text):
        result = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
        )
        return result.data[0].embedding

    def summarize(self, text):
        system_prompt = (
            "summarize the following text in 2-3 sentences"
        )
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
