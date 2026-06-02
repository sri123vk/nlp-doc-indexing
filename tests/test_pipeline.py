from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aidoc.evaluation.evaluator import EvaluationQuestion, RetrievalEvaluator
from aidoc.pipeline import DocumentIntelligencePipeline


class PipelineTest(unittest.TestCase):
    def test_pipeline_indexes_retrieves_and_builds_citations(self) -> None:
        with TemporaryDirectory() as directory:
            docs = Path(directory) / "docs"
            docs.mkdir()
            (docs / "policy.txt").write_text(
                "Page 1 of 1\n"
                "The control owner must collect audit evidence. "
                "Critical exceptions require remediation and executive acknowledgement.",
                encoding="utf-8",
            )

            pipeline = DocumentIntelligencePipeline()
            summary = pipeline.index_path(docs)
            answer = pipeline.ask("What requires remediation?")

            self.assertEqual(summary["documents"], 1)
            self.assertGreaterEqual(summary["chunks"], 1)
            self.assertEqual(summary["enriched_chunks"], summary["chunks"])
            self.assertGreaterEqual(summary["entities"], 1)
            self.assertGreaterEqual(summary["relations"], 1)
            self.assertGreaterEqual(summary["knowledge_graph_triples"], 1)
            self.assertTrue(answer["citations"])
            self.assertIn("remediation", answer["answer"].lower())

            analysis = pipeline.query_understanding.analyze("Which vendor policy discusses termination and data return?")
            self.assertEqual(analysis.intent, "retrieval")
            self.assertEqual(analysis.domain_hint, "vendor_risk")
            self.assertTrue(any(token.pos == "NOUN" for token in analysis.tokens))

            evaluation = RetrievalEvaluator(pipeline).evaluate([
                EvaluationQuestion(
                    question="What requires remediation?",
                    expected_terms=["remediation"],
                    expected_document_hint="policy",
                )
            ])
            self.assertEqual(evaluation.questions, 1)
            self.assertEqual(evaluation.recall_at_k, 1.0)


if __name__ == "__main__":
    unittest.main()
