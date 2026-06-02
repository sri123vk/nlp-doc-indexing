from __future__ import annotations

from aidoc.models import ChunkEnrichment, QueryAnalysis, RetrievedChunk


class QueryAwareReranker:
    def rerank(
        self,
        query_analysis: QueryAnalysis,
        candidates: list[RetrievedChunk],
        enrichments: list[ChunkEnrichment],
    ) -> list[RetrievedChunk]:
        enrichment_by_chunk = {item.chunk.chunk_id: item for item in enrichments}
        reranked: list[RetrievedChunk] = []

        for candidate in candidates:
            enrichment = enrichment_by_chunk.get(candidate.chunk.chunk_id)
            rerank_score = candidate.final_score
            text = candidate.chunk.text.lower()

            for token in query_analysis.tokens:
                if token.text in text:
                    rerank_score += 0.04
                if token.entity_label and enrichment and any(entity.label == token.entity_label for entity in enrichment.entities):
                    rerank_score += 0.08

            for entity in query_analysis.entities:
                if entity.text in text:
                    rerank_score += 0.12

            for phrase in query_analysis.keyphrases:
                if phrase.text in text:
                    rerank_score += 0.06

            if enrichment:
                if enrichment.document_label.label == query_analysis.domain_hint:
                    rerank_score += 0.18
                if enrichment.risk_label.severity in {"high", "critical"} and query_analysis.intent in {"risk_review", "retrieval"}:
                    rerank_score += 0.1
                if any(relation.relation == "requires" for relation in enrichment.relations):
                    rerank_score += 0.05

            reranked.append(
                RetrievedChunk(
                    chunk=candidate.chunk,
                    lexical_score=candidate.lexical_score,
                    vector_score=candidate.vector_score,
                    final_score=candidate.final_score,
                    rerank_score=rerank_score,
                )
            )

        return sorted(reranked, key=lambda item: item.rerank_score, reverse=True)

