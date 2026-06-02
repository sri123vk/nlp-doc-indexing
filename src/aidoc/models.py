from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RawDocument:
    doc_id: str
    title: str
    text: str
    source_uri: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_uri: str
    page: int | None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Entity:
    text: str
    label: str
    confidence: float


@dataclass(frozen=True)
class Keyphrase:
    text: str
    score: float


@dataclass(frozen=True)
class DocumentLabel:
    label: str
    confidence: float
    rationale: str


@dataclass(frozen=True)
class RiskLabel:
    label: str
    severity: str
    confidence: float
    rationale: str


@dataclass(frozen=True)
class Relation:
    subject: str
    relation: str
    object: str
    confidence: float


@dataclass(frozen=True)
class ChunkEnrichment:
    chunk: Chunk
    entities: list[Entity]
    keyphrases: list[Keyphrase]
    document_label: DocumentLabel
    risk_label: RiskLabel
    relations: list[Relation]
    summary: str


@dataclass(frozen=True)
class QueryToken:
    text: str
    pos: str
    entity_label: str | None = None


@dataclass(frozen=True)
class QueryAnalysis:
    query: str
    normalized_query: str
    tokens: list[QueryToken]
    entities: list[Entity]
    keyphrases: list[Keyphrase]
    intent: str
    domain_hint: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    lexical_score: float
    vector_score: float
    final_score: float
    rerank_score: float = 0.0


@dataclass(frozen=True)
class EntityTriple:
    subject: str
    relation: str
    object: str
    chunk_id: str
    source_uri: str
