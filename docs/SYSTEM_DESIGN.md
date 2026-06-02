# NLP DOC Indexing System Design

## Phase Boundary

This repository is the AI intelligence layer. It assumes raw text is available from a traditional ingestion system and focuses on chunk level retrieval, knowledge extraction, and citation grounded responses.

## Components

| Component | Responsibility |
| :-- | :-- |
| Loader | Reads source documents |
| Chunker | Preserves page level context and creates retrievable units |
| Foundation model provider | Abstracts embeddings, summarization, classification, keyphrases, and risk labeling |
| NLP enrichment pipeline | Extracts entities, keyphrases, document labels, risk labels, summaries, and relations |
| Query understanding | Extracts query intent, entities, keyphrases, and simple POS tags |
| Embedding model | Converts chunk text into vectors |
| Hybrid retriever | Combines lexical and semantic evidence |
| Query reranker | Reorders evidence using query and enrichment signals |
| Knowledge graph extractor | Converts unstructured text into relationship triples |
| Answer builder | Produces responses with citations |
| Evaluation harness | Measures retrieval recall, citation hit rate, matched terms, and average score |

## Implemented Local Prototype

The local prototype includes document loading, semantic chunking, deterministic foundation model fallback, NLP enrichment, query understanding, reranking, hybrid retrieval, citation answer generation, knowledge graph triple extraction, and evaluation.

Verified corpus output:

```text
documents: 5
chunks: 305
enriched_chunks: 305
knowledge_graph_triples: 3976
entities: 2566
relations: 800
```

## Production Upgrade Path

1. Replace local text loading with S3 or Lucene metadata integration
2. Replace local foundation model provider with transformer backends
3. Add GPU embedding and reranking workers
4. Add a persistent vector index
5. Add cross encoder reranking
6. Add LLM answer generation
7. Store knowledge graph triples in Neo4j or Amazon Neptune
8. Add evaluation metrics for retrieval quality and citation faithfulness

## NLP And Foundation Model Tasks

The current code models the following foundation model tasks through local deterministic implementations:

| Task | Current Implementation | Production Upgrade |
| :-- | :-- | :-- |
| Embeddings | Hash based dense vectors | Sentence transformer, OpenAI, or Bedrock |
| Summarization | Extractive sentence summary | Instruction tuned LLM |
| Classification | Domain keyword classifier | Fine tuned transformer classifier |
| Risk labeling | Pattern based severity labeling | Compliance risk model |
| Keyphrases | Frequency based extraction | Sequence to sequence or embedding based extraction |
| NER | Domain pattern extractor | Transformer token classifier |
| Relation extraction | Rule based relations | LLM or supervised relation extractor |

## GPU Plan

GPU workers should be isolated behind model interfaces:

```text
chunk batch -> GPU embedding worker -> vector store
retrieved chunks -> GPU reranker -> final evidence
document pages -> GPU OCR worker -> extracted text
```

This lets the CPU based indexing and metadata layer scale independently from model inference workloads.
