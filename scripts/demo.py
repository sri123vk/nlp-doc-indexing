from __future__ import annotations

import json
from pathlib import Path

from aidoc.pipeline import DocumentIntelligencePipeline


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "sample-docs" / "compliance"
    pipeline = DocumentIntelligencePipeline()
    summary = pipeline.index_path(root)
    answer = pipeline.ask("Which policy evidence mentions remediation and executive acknowledgement?")

    print("Index summary")
    print(json.dumps(summary, indent=2))
    print()
    print("RAG style answer")
    print(json.dumps(answer, indent=2))
    print()
    print("NLP enrichment sample")
    sample = pipeline.enrichments[0]
    print(json.dumps({
        "chunk_id": sample.chunk.chunk_id,
        "document_label": sample.document_label.__dict__,
        "risk_label": sample.risk_label.__dict__,
        "summary": sample.summary,
        "entities": [entity.__dict__ for entity in sample.entities[:8]],
        "keyphrases": [phrase.__dict__ for phrase in sample.keyphrases[:8]],
        "relations": [relation.__dict__ for relation in sample.relations[:8]],
    }, indent=2))
    print()
    print("Query understanding sample")
    analysis = pipeline.query_understanding.analyze("Which vendor policy discusses termination and data return?")
    print(json.dumps({
        "query": analysis.query,
        "normalized_query": analysis.normalized_query,
        "intent": analysis.intent,
        "domain_hint": analysis.domain_hint,
        "tokens": [token.__dict__ for token in analysis.tokens],
        "entities": [entity.__dict__ for entity in analysis.entities],
        "keyphrases": [phrase.__dict__ for phrase in analysis.keyphrases],
    }, indent=2))
    print()
    print("Knowledge graph sample")
    print(json.dumps([triple.__dict__ for triple in pipeline.triples[:8]], indent=2))


if __name__ == "__main__":
    main()
