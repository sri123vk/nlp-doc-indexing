# NLP DOC Indexing

NLP DOC Indexing is a foundation model and NLP based document intelligence engine. It is the second project in a two part document search system. The first project, Lucene Scalable DocSearch Index Engine, handles traditional inverted indexing, BM25 search, sharding, streaming updates, and metadata checkpoints. This project handles the AI layer: semantic chunking, NLP enrichment, hybrid retrieval, citation grounded answers, knowledge graph construction, and retrieval evaluation.

The repository is intentionally runnable without paid APIs or external model servers. It ships with deterministic local model backends so the full pipeline can be tested on any machine. The architecture still follows the same interface boundaries used by production foundation model systems, so OpenAI, Amazon Bedrock, Hugging Face, vLLM, Ollama, or GPU backed sentence transformer backends can replace the local implementations later.

## 1. Completed System

The project is no longer only a RAG sketch. It now contains a complete local document intelligence pipeline:

| Layer | Implemented Capability |
| :-- | :-- |
| Document ingestion | Loads compliance text documents from a local corpus |
| Semantic chunking | Splits page aware text into overlapping retrieval units |
| Foundation model interface | Defines embeddings, summarization, classification, keyphrase extraction, and risk labeling APIs |
| Local model backend | Provides deterministic embeddings and NLP outputs without external services |
| NLP enrichment | Extracts entities, keyphrases, document labels, risk labels, summaries, and relations |
| Query understanding | Tags intent, entities, keyphrases, and simple POS categories |
| Hybrid retrieval | Combines BM25 style lexical scoring with vector similarity |
| Query reranking | Reorders evidence using query and enrichment signals |
| RAG answer builder | Produces evidence based answers with chunk level citations |
| Knowledge graph | Converts enriched chunks into entity and relationship triples |
| Evaluation harness | Measures recall at K, citation hit rate, matched terms, and average score |
| CI | Runs the test suite through GitHub Actions |

## 2. Research Question

The project answers this question:

```text
How can enterprise documents be transformed into semantic chunks, structured NLP signals, retrievable evidence, knowledge graph triples, and citation grounded answers using a foundation model ready architecture?
```

## 3. Architecture

```text
Compliance Documents
        |
        v
Document Loader
        |
        v
Semantic Chunker
        |
        v
NLP Enrichment Pipeline
        |
        +--> Entity Extraction
        +--> Keyphrase Extraction
        +--> Document Classification
        +--> Risk Labeling
        +--> Relation Extraction
        +--> Summarization
        |
        v
Query Understanding
        |
        v
Foundation Model Provider Interface
        |
        +--> Local deterministic backend
        +--> Future OpenAI or Bedrock backend
        +--> Future GPU transformer backend
        |
        v
Hybrid Retriever
        |
        +--> BM25 style lexical score
        +--> Vector cosine score
        |
        v
Citation Grounded Answer
```

Knowledge graph construction runs in parallel:

```text
Enriched Chunks
        |
        v
Entity and Relation Projection
        |
        v
Knowledge Graph Triples
```

## 4. Verified Demo Output

Running the demo over the compliance corpus produces:

```text
documents: 5
chunks: 305
enriched_chunks: 305
knowledge_graph_triples: 3976
entities: 2566
relations: 800
```

The RAG style answer returns citations with:

```text
chunk_id
document title
source URI
page number
retrieval score
```

The NLP enrichment sample includes:

```text
document_label
risk_label
summary
entities
keyphrases
relations
```

The query understanding layer adds:

```text
intent
domain hint
simple POS tags
query entities
query keyphrases
```

## 5. Run The Project

```bash
cd /Users/srimathiravisankar/nlp-doc-indexing
PYTHONPATH=src python3 scripts/demo.py
```

Or:

```bash
cd /Users/srimathiravisankar/nlp-doc-indexing
chmod +x scripts/demo.sh
./scripts/demo.sh
```

## 6. Run The Evaluation Harness

```bash
cd /Users/srimathiravisankar/nlp-doc-indexing
PYTHONPATH=src python3 scripts/evaluate.py
```

The evaluator reads curated questions from:

```text
data/evaluation/compliance_questions.json
```

It reports:

| Metric | Meaning |
| :-- | :-- |
| `recall_at_k` | Whether expected terms appeared in retrieved evidence |
| `citation_hit_rate` | Whether returned citations came from the expected document |
| `average_score` | Mean hybrid retrieval score across evidence |
| `matched_terms` | Expected terms found in retrieved chunks |

Verified local evaluation:

```text
questions: 4
recall_at_k: 1.0
citation_hit_rate: 1.0
average_score: 4.6411
```

## 7. Run Tests

```bash
cd /Users/srimathiravisankar/nlp-doc-indexing
PYTHONPATH=src python3 -m unittest discover -s tests
```

Verified locally:

```text
Ran 1 test
OK
```

## 8. Core Modules

| Module | Responsibility |
| :-- | :-- |
| `aidoc.chunking.SemanticChunker` | Page aware overlapping chunk generation |
| `aidoc.foundation.FoundationModelProvider` | Model backend abstraction |
| `aidoc.foundation.LocalFoundationModelProvider` | Local deterministic model backend |
| `aidoc.nlp.NLPEnrichmentPipeline` | Entity, keyphrase, label, risk, relation, and summary generation |
| `aidoc.nlp.QueryUnderstanding` | Query intent, entity, keyphrase, and POS tagging |
| `aidoc.retrieval.HybridRetriever` | BM25 style and vector retrieval fusion |
| `aidoc.reranking.QueryAwareReranker` | Query aware evidence reranking |
| `aidoc.kg.EnrichedKnowledgeGraphExtractor` | Knowledge graph triple construction |
| `aidoc.rag.CitationAnswerBuilder` | Citation grounded answer construction |
| `aidoc.evaluation.RetrievalEvaluator` | Retrieval and citation quality evaluation |

## 9. Foundation Model Design

The project uses a provider interface instead of hard coding one vendor. The current local backend implements the same tasks expected from a production foundation model system:

| Task | Local Implementation | Production Replacement |
| :-- | :-- | :-- |
| Embeddings | Hash based dense vector | OpenAI, Bedrock, sentence transformers |
| Summarization | Extractive summary | Instruction tuned LLM |
| Classification | Domain keyword classifier | Fine tuned transformer classifier |
| Risk labeling | Pattern based severity model | Compliance risk classifier |
| Keyphrases | Frequency based phrases | LLM or embedding based extraction |
| Entity extraction | Domain pattern NER | Transformer token classifier |
| Relation extraction | Rule based relations | LLM or supervised relation extractor |

This makes the repo credible as a foundation model system while remaining fully runnable in a local academic environment.

## 10. Query Understanding And Reranking

The repository includes a query understanding layer that extracts:

1. Query intent
2. Query entities
3. Query keyphrases
4. Simple POS tags
5. Domain hints for privacy, financial controls, security, vendor risk, and AI governance

The reranker then reorders candidate chunks using:

1. Lexical overlap with the query
2. Entity label alignment
3. Keyphrase overlap
4. Domain label agreement
5. Risk label awareness when the query is about risk

This is the right insertion point for a future transformer reranker or cross encoder.

## 11. GPU Story

GPU acceleration belongs in the model inference layer. It is not needed for the deterministic local fallback, but the architecture is ready for GPU backed upgrades:

| GPU Component | Role |
| :-- | :-- |
| Embedding worker | Batch transformer embedding generation |
| Reranker | Cross encoder scoring of query and chunk pairs |
| OCR worker | Scanned PDF and image text extraction |
| NER model | Token classification over document chunks |
| Summarization model | Long context summary generation |
| Local LLM server | RAG answer generation through vLLM or Ollama |

Recommended production path:

```text
chunks
   |
   v
GPU embedding worker
   |
   v
vector index
   |
   v
hybrid retrieval
   |
   v
GPU reranker
   |
   v
LLM answer generator
   |
   v
citation verifier
```

## 12. Relationship To The Lucene Project

The Lucene project answers:

```text
How do we index and search documents at scale using traditional information retrieval?
```

This project answers:

```text
How do we understand, enrich, retrieve, cite, and reason over those documents using NLP and foundation model concepts?
```

Together:

```text
Lucene Scalable DocSearch Index Engine
        +
NLP DOC Indexing
        =
Scalable Enterprise Document Intelligence Platform
```

## 13. Project Status

The current repository is complete as a local, testable masters level prototype. It includes implemented code, sample data, a demo, tests, a CI workflow, a system design document, a model abstraction layer, NLP enrichment, query understanding, reranking, hybrid retrieval, knowledge graph construction, RAG style answers, and an evaluation harness.

Future work should focus on replacing local deterministic models with production model backends rather than changing the architecture.
