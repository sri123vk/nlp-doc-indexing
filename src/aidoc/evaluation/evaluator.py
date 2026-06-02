from __future__ import annotations

from dataclasses import dataclass

from aidoc.pipeline import DocumentIntelligencePipeline


@dataclass(frozen=True)
class EvaluationQuestion:
    question: str
    expected_terms: list[str]
    expected_document_hint: str


@dataclass(frozen=True)
class EvaluationResult:
    question: str
    retrieved: bool
    citation_hit: bool
    average_score: float
    matched_terms: list[str]


@dataclass(frozen=True)
class EvaluationSummary:
    questions: int
    recall_at_k: float
    citation_hit_rate: float
    average_score: float
    results: list[EvaluationResult]


class RetrievalEvaluator:
    def __init__(self, pipeline: DocumentIntelligencePipeline) -> None:
        self.pipeline = pipeline

    def evaluate(self, questions: list[EvaluationQuestion], limit: int = 5) -> EvaluationSummary:
        results = [self._evaluate_question(question, limit) for question in questions]
        total = max(1, len(results))
        return EvaluationSummary(
            questions=len(results),
            recall_at_k=sum(1 for result in results if result.retrieved) / total,
            citation_hit_rate=sum(1 for result in results if result.citation_hit) / total,
            average_score=sum(result.average_score for result in results) / total,
            results=results,
        )

    def _evaluate_question(self, question: EvaluationQuestion, limit: int) -> EvaluationResult:
        evidence = self.pipeline.retriever.search(question.question, limit=limit)
        answer = self.pipeline.answer_builder.build(question.question, evidence)
        combined_text = " ".join(hit.chunk.text.lower() for hit in evidence)
        matched_terms = [
            term for term in question.expected_terms
            if term.lower() in combined_text
        ]
        citation_hit = any(
            question.expected_document_hint.lower() in citation["source_uri"].lower()
            for citation in answer["citations"]
        )
        return EvaluationResult(
            question=question.question,
            retrieved=len(matched_terms) > 0,
            citation_hit=citation_hit,
            average_score=sum(hit.final_score for hit in evidence) / max(1, len(evidence)),
            matched_terms=matched_terms,
        )

