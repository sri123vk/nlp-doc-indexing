from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from aidoc.evaluation.evaluator import EvaluationQuestion, RetrievalEvaluator
from aidoc.pipeline import DocumentIntelligencePipeline


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    corpus_root = project_root / "sample-docs" / "compliance"
    questions_path = project_root / "data" / "evaluation" / "compliance_questions.json"

    questions = [
        EvaluationQuestion(
            question=item["question"],
            expected_terms=item["expected_terms"],
            expected_document_hint=item["expected_document_hint"],
        )
        for item in json.loads(questions_path.read_text(encoding="utf-8"))
    ]

    pipeline = DocumentIntelligencePipeline()
    pipeline.index_path(corpus_root)
    summary = RetrievalEvaluator(pipeline).evaluate(questions)
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()

