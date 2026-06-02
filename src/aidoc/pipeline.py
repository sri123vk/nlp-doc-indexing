from __future__ import annotations

from pathlib import Path

from aidoc.chunking.semantic_chunker import SemanticChunker
from aidoc.foundation.model_provider import FoundationModelProvider, LocalFoundationModelProvider
from aidoc.io.local_loader import load_text_documents
from aidoc.kg.knowledge_graph import EnrichedKnowledgeGraphExtractor
from aidoc.nlp.enrichment import NLPEnrichmentPipeline
from aidoc.nlp.query_understanding import QueryUnderstanding
from aidoc.rag.answer_builder import CitationAnswerBuilder
from aidoc.reranking.query_reranker import QueryAwareReranker
from aidoc.retrieval.hybrid_retriever import HybridRetriever


class DocumentIntelligencePipeline:
    def __init__(self, model_provider: FoundationModelProvider | None = None) -> None:
        self.model_provider = model_provider or LocalFoundationModelProvider()
        self.chunker = SemanticChunker()
        self.retriever = HybridRetriever(self.model_provider)
        self.enrichment_pipeline = NLPEnrichmentPipeline(self.model_provider)
        self.query_understanding = QueryUnderstanding()
        self.reranker = QueryAwareReranker()
        self.graph_extractor = EnrichedKnowledgeGraphExtractor()
        self.answer_builder = CitationAnswerBuilder()
        self.chunks = []
        self.enrichments = []
        self.triples = []

    def index_path(self, root: Path) -> dict[str, int]:
        documents = load_text_documents(root)
        self.chunks = [chunk for document in documents for chunk in self.chunker.chunk(document)]
        self.enrichments = self.enrichment_pipeline.enrich(self.chunks)
        self.retriever.index(self.chunks)
        self.triples = self.graph_extractor.extract(self.enrichments)
        return {
            "documents": len(documents),
            "chunks": len(self.chunks),
            "enriched_chunks": len(self.enrichments),
            "knowledge_graph_triples": len(self.triples),
            "entities": sum(len(enrichment.entities) for enrichment in self.enrichments),
            "relations": sum(len(enrichment.relations) for enrichment in self.enrichments),
        }

    def ask(self, question: str, limit: int = 5) -> dict[str, object]:
        query_analysis = self.query_understanding.analyze(question)
        evidence = self.retriever.search(query_analysis.normalized_query, limit=max(10, limit * 2))
        evidence = self.reranker.rerank(query_analysis, evidence, self.enrichments)[:limit]
        return self.answer_builder.build(question, evidence)
