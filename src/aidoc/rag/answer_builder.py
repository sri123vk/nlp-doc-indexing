from __future__ import annotations

from aidoc.models import RetrievedChunk


class CitationAnswerBuilder:
    def build(self, question: str, evidence: list[RetrievedChunk]) -> dict[str, object]:
        if not evidence:
            return {
                "question": question,
                "answer": "No relevant evidence was retrieved from the indexed corpus.",
                "citations": [],
            }

        top = evidence[:3]
        answer_sentences = [
            "The retrieved compliance evidence indicates the following:",
        ]
        for index, hit in enumerate(top, start=1):
            preview = hit.chunk.text[:260].strip()
            answer_sentences.append(f"{index}. {preview}")

        citations = [
            {
                "chunk_id": hit.chunk.chunk_id,
                "title": hit.chunk.title,
                "source_uri": hit.chunk.source_uri,
                "page": hit.chunk.page,
                "score": round(hit.rerank_score if hit.rerank_score else hit.final_score, 4),
            }
            for hit in top
        ]

        return {
            "question": question,
            "answer": " ".join(answer_sentences),
            "citations": citations,
        }
