# NLP DOC Indexing

NLP DOC Indexing is the second phase of the document search project. The first repository built the traditional Lucene foundation with batch indexing, streaming updates, physical shards, checkpointing, and BM25 search. This repository focuses on foundation models and NLP for document indexing: semantic chunking, embeddings, hybrid retrieval, entity extraction, keyphrase extraction, document classification, risk labeling, relation extraction, summarization, citation grounded answer generation, and knowledge graph construction.

The first version is intentionally runnable without external model APIs. It uses deterministic local hash embeddings so the architecture, tests, and demo work on any machine. The model interfaces are designed so a production embedding model, GPU backed transformer, reranker, or managed LLM can replace the local implementation later.

## 1. Research Goal

The central question is:

How can a classical document search system be extended into a document intelligence platform that retrieves evidence, builds citations, extracts relationships, and prepares for retrieval augmented generation?

This repository separates AI retrieval from the Lucene infrastructure repo so each system has a clear responsibility.

## 2. Current Capabilities

| Capability | Implementation |
| :-- | :-- |
| Document loading | Local text document loader |
| Semantic chunking | Page aware and overlap aware chunker |
| Embeddings | Deterministic local hash embedding model |
| Foundation model abstraction | Swappable provider for embeddings, summarization, classification, and risk labeling |
| Entity extraction | Rule based NER for compliance roles, controls, risks, artifacts, and AI governance terms |
| Keyphrase extraction | Frequency based local keyphrase extraction |
| Document classification | Local classifier for privacy, financial controls, security, vendor risk, and AI governance |
| Risk labeling | Severity labels for low, medium, high, and critical compliance risk |
| Relation extraction | Extracted relationships such as exception requires remediation |
| Summarization | Local extractive summary fallback |
| Lexical retrieval | BM25 style scoring over chunks |
| Vector retrieval | Cosine similarity over local embeddings |
| Hybrid retrieval | Weighted lexical and vector score fusion |
| RAG answer | Evidence based answer builder with citations |
| Knowledge graph | Rule based entity and relationship extraction |
| Demo corpus | Compliance documents copied from the Lucene project |
| Tests | Standard library unittest pipeline coverage |

## 3. Architecture

```text
Documents
        |
        v
Semantic Chunker
        |
        v
NLP Enrichment
        |
        v
Entities, Keyphrases, Labels, Risks, Relations, Summaries
        |
        v
Foundation Model Interface
        |
        +-------------------+
        |                   |
        v                   v
BM25 Scoring          Local Embeddings
        |                   |
        +---------+---------+
                  |
                  v
           Hybrid Retrieval
                  |
                  v
        Citation Answer Builder
                  |
                  v
          RAG Style Response
```

The knowledge graph path runs beside retrieval:

```text
Chunks
   |
   v
Entity Extraction
   |
   v
Relationship Extraction
   |
   v
Knowledge Graph Triples
```

## 4. Run the Demo

```bash
cd /Users/srimathiravisankar/ai-rag-docsearch-knowledge-engine
chmod +x scripts/demo.sh
./scripts/demo.sh
```

The demo indexes the compliance corpus, retrieves evidence for a compliance question, returns citations, and prints sample knowledge graph triples.

## 5. Run Tests

```bash
cd /Users/srimathiravisankar/ai-rag-docsearch-knowledge-engine
PYTHONPATH=src python3 -m unittest discover -s tests
```

## 6. Core Modules

| Path | Role |
| :-- | :-- |
| `SemanticChunker` | Splits documents into page aware overlapping chunks |
| `FoundationModelProvider` | Defines model backed embedding, classification, keyphrase, summary, and risk APIs |
| `LocalFoundationModelProvider` | Deterministic local fallback for tests and demos |
| `NLPEnrichmentPipeline` | Runs entity extraction, keyphrase extraction, classification, risk labeling, relation extraction, and summarization |
| `EntityExtractor` | Extracts compliance and AI governance entities |
| `RelationExtractor` | Extracts domain relationships from enriched chunks |
| `LocalHashEmbeddingModel` | Provides deterministic local vectors for reproducible demos |
| `HybridRetriever` | Combines BM25 style scoring with vector similarity |
| `KnowledgeGraphExtractor` | Extracts entity relationship triples from chunks |
| `CitationAnswerBuilder` | Builds answer payloads with chunk level citations |
| `DocumentIntelligencePipeline` | Orchestrates loading, chunking, indexing, graph extraction, retrieval, and answer generation |

## 7. GPU Roadmap

GPU acceleration belongs in the model inference path, not in the classical Lucene inverted index. This repository is the correct place to add it.

GPU suitable additions:

| Component | GPU Value |
| :-- | :-- |
| Embedding generation | Transformer inference can be batched across chunks |
| OCR for scanned PDFs | Vision text recognition benefits from parallel computation |
| Cross encoder reranking | Query chunk pairs can be scored in batches |
| Entity extraction | Token classification can run on GPU |
| Summarization | Long document summarization uses transformer decoding |

The future production path should replace `LocalHashEmbeddingModel` with one of the following:

1. A local sentence transformer running on GPU
2. Amazon Bedrock embeddings
3. OpenAI embeddings
4. A managed vector search and reranking service

## 8. Foundation Model Roadmap

The local provider is intentionally deterministic. A 2026 production version should add provider implementations for:

| Provider | Use |
| :-- | :-- |
| Hugging Face sentence transformers | Local embedding generation and reranking |
| OpenAI embeddings and responses | Managed embeddings, summarization, and grounded answer generation |
| Amazon Bedrock | Enterprise managed embeddings and LLM inference |
| vLLM or Ollama | GPU backed local LLM serving |
| Cross encoder reranker | More accurate evidence ordering after retrieval |

The recommended model pipeline is:

```text
chunks
   |
   v
embedding model
   |
   v
vector retrieval
   |
   v
cross encoder reranker
   |
   v
grounded answer model
   |
   v
verifier requiring citations
```

## 9. Relationship To The Lucene Repository

The Lucene repository answers:

```text
How do we index and search documents at scale using traditional IR?
```

This repository answers:

```text
How do we understand, retrieve, cite, and reason over those documents?
```

Together they form a complete document intelligence platform:

```text
Lucene Search Infrastructure
        +
AI RAG Knowledge Engine
        =
Scalable Enterprise Document Intelligence
```
