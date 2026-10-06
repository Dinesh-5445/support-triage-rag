# Support Triage RAG

An advanced modular **Retrieval-Augmented Generation (RAG) system for
support-ticket triage**, designed around evidence retrieval, grounding,
safety, bounded reflection, and deterministic decision-making.

This repository is the architectural evolution of the original **Support
Triage Agent**. The original project remains separate as the
deterministic NLP/TF-IDF system; this repository contains the advanced
modular RAG implementation.

> **Current scope:** This is an evidence-first RAG backend. It does not
> currently use an external LLM API for response generation and does not
> include a frontend, FastAPI service, cloud deployment, conversation
> memory, GraphRAG, a vector database, or an autonomous agent loop.

------------------------------------------------------------------------

## Table of Contents

-   [Overview](#overview)
-   [Goals](#goals)
-   [Architecture](#architecture)
-   [End-to-End Flow](#end-to-end-flow)
-   [Knowledge Ingestion](#knowledge-ingestion)
-   [Query Processing](#query-processing)
-   [Safety and Domain Gate](#safety-and-domain-gate)
-   [Retrieval](#retrieval)
-   [Ranking and Context](#ranking-and-context)
-   [Grounding](#grounding)
-   [Deterministic Response
    Generation](#deterministic-response-generation)
-   [Reflection and Retry](#reflection-and-retry)
-   [Final Decision](#final-decision)
-   [Module Responsibilities](#module-responsibilities)
-   [Repository Structure](#repository-structure)
-   [Evaluation](#evaluation)
-   [Results](#results)
-   [Windows Semantic Retrieval
    Limitation](#windows-semantic-retrieval-limitation)
-   [Testing](#testing)
-   [Installation and Usage](#installation-and-usage)
-   [Design Decisions](#design-decisions)
-   [Explicit Non-Goals](#explicit-non-goals)
-   [Relationship to the Original
    Project](#relationship-to-the-original-project)
-   [Limitations](#limitations)
-   [Future Direction](#future-direction)
-   [Research Reference](#research-reference)

------------------------------------------------------------------------

## Overview

The system treats support triage as an **evidence and decision
problem**, not simply as:

``` text
Ticket → LLM → Answer
```

Instead, an incoming ticket passes through:

``` text
Support Ticket
     │
     ▼
Query Analysis
     │
     ▼
Query Rewriting
     │
     ▼
Safety / Domain Gate
     │
     ▼
Retrieval Router
     │
     ├── TF-IDF
     ├── Dense Semantic Retrieval
     └── Hybrid Retrieval
              │
              ▼
        Score Fusion
              │
              ▼
     Score-Based Reranking
              │
              ▼
       Context Filtering
              │
              ▼
      Evidence Grounding
              │
              ▼
     Bounded Reflection
              │
              ▼
 Deterministic Response Extraction
              │
              ▼
      Final Safety Decision
              │
      ┌───────┼───────────────┐
      ▼       ▼               ▼
   ANSWER  PARTIAL+ESCALATE  ESCALATE
              │
              └── OUT_OF_SCOPE is handled by the domain gate
```

The architecture separates retrieval, ranking, evidence validation,
reflection, generation, and safety so that each stage can be evaluated
independently.

------------------------------------------------------------------------

# Goals

The system is designed around five goals:

1.  **Evidence-first answering** --- responses should be supported by
    retrieved documentation.
2.  **Modular retrieval** --- lexical, semantic, and hybrid retrieval
    should be interchangeable.
3.  **Explicit grounding** --- retrieved evidence is classified as
    `GROUNDED`, `PARTIAL`, or `UNGROUNDED`.
4.  **Safety-aware automation** --- risky or unsupported cases are
    escalated instead of being forced into answers.
5.  **Bounded recovery** --- insufficient evidence can trigger a limited
    retry rather than an uncontrolled loop.

------------------------------------------------------------------------

# Architecture

## High-Level Architecture

``` mermaid
flowchart LR
    A["Support Knowledge Base"] --> B["Loader"]
    B --> C["Cleaning"]
    C --> D["Chunking<br/>300 words / 50 overlap"]
    D --> E["Indexes"]

    F["Support Ticket"] --> G["Query Analyzer"]
    G --> H["Query Rewriter"]
    H --> I["Safety / Domain Gate"]
    I --> J["Retrieval Router"]

    E --> K["TF-IDF Index"]
    E --> L["Dense Index"]
    J --> K
    J --> L

    K --> M["Score Fusion"]
    L --> M
    M --> N["Score-Based Reranking"]
    N --> O["Context Filter<br/>Top 3"]
    O --> P["Grounding"]

    P --> Q{"Grounding State"}
    Q -->|"GROUNDED"| R["Deterministic Response"]
    Q -->|"PARTIAL / UNGROUNDED"| S["Bounded Retry"]

    S -->|"Attempts < 2"| J
    S -->|"Retry limit reached"| T["ESCALATE"]

    R --> U["Final Safety / Decision"]
    U --> V["ANSWER"]
    U --> W["PARTIAL_ANSWER_ESCALATE"]
    U --> T

    I -->|"High Risk"| T
    I -->|"Out of Scope"| X["OUT_OF_SCOPE"]
```

## Layered View

``` text
┌──────────────────────────────────────────────────────────────┐
│ KNOWLEDGE LAYER                                              │
│ Markdown → Cleaning → Chunking → TF-IDF / Dense Indexes      │
├──────────────────────────────────────────────────────────────┤
│ QUERY LAYER                                                  │
│ Ticket → Analysis → Rewrite → Retrieval Routing              │
├──────────────────────────────────────────────────────────────┤
│ RETRIEVAL LAYER                                              │
│ TF-IDF / Dense / Hybrid → Fusion → Score-Based Reranking      │
├──────────────────────────────────────────────────────────────┤
│ CONTEXT LAYER                                                │
│ Deduplication → Filtering → Top-3 Evidence                    │
├──────────────────────────────────────────────────────────────┤
│ GROUNDING / REFLECTION                                       │
│ Evidence Check → Grounding State → Bounded Retry             │
├──────────────────────────────────────────────────────────────┤
│ DECISION LAYER                                               │
│ Extractive Response → Safety → Answer / Escalation            │
└──────────────────────────────────────────────────────────────┘
```

------------------------------------------------------------------------

# End-to-End Flow

``` mermaid
flowchart TD
    A["Incoming Support Ticket"] --> B["Query Analyzer"]
    B --> C["Query Rewriter"]
    C --> D{"Safety / Domain Gate"}

    D -->|"High Risk"| E["ESCALATE"]
    D -->|"Out of Scope"| F["OUT_OF_SCOPE"]
    D -->|"Allowed"| G["Retrieval Router"]

    G --> H["TF-IDF"]
    G --> I["Dense"]
    G --> J["Hybrid"]

    H --> K["Candidate Evidence"]
    I --> K
    J --> K

    K --> L["Score Fusion"]
    L --> M["Score-Based Reranking"]
    M --> N["Context Filter"]
    N --> O["Grounding Evaluator"]

    O --> P{"Evidence State"}
    P -->|"GROUNDED"| Q["Deterministic Response"]
    P -->|"PARTIAL"| R["Retry Policy"]
    P -->|"UNGROUNDED"| R

    R --> S{"Attempts < 2?"}
    S -->|"Yes"| G
    S -->|"No"| E

    Q --> T["Final Decision"]
    T --> U["ANSWER"]
    T --> V["PARTIAL_ANSWER_ESCALATE"]
    T --> E
```

------------------------------------------------------------------------

# Knowledge Ingestion

The ingestion layer converts the local support corpus into a common
retrieval representation.

``` mermaid
flowchart LR
    A["Markdown Support Documents"] --> B["Document Loader"]
    B --> C["Cleaning / Normalization"]
    C --> D["Chunker"]
    D --> E["300-word Chunks"]
    E --> F["50-word Overlap"]
    F --> G["Indexer"]
    G --> H["TF-IDF Index"]
    G --> I["Dense Embeddings"]
```

### Chunk configuration

``` text
Chunk size:     300 words
Overlap:         50 words
```

The overlap reduces information loss when a relevant passage crosses a
chunk boundary.

The knowledge base currently contains:

``` text
data/
├── hackerrank/
├── claude/
└── visa/
```

------------------------------------------------------------------------

# Query Processing

## Query Analyzer

The analyzer exposes deterministic information about the incoming
ticket, including:

-   request type,
-   product/domain,
-   relevant query signals,
-   risk indicators.

The classification logic is derived from the earlier deterministic
support-triage implementation but is now separated as a reusable query
module.

## Query Rewriter

The V2 rewriter is intentionally lightweight.

It does not call an LLM. It performs an identity-style normalization and
token expansion so the retrieval layer receives a more useful
representation without introducing uncontrolled generated text.

## Retrieval Router

The router provides a common interface for:

``` text
TF-IDF
Dense
Hybrid
```

This keeps retrieval strategy separate from the rest of the pipeline.

------------------------------------------------------------------------

# Safety and Domain Gate

Safety is checked before retrieval and again after evidence processing.

``` mermaid
flowchart TD
    A["Ticket"] --> B{"Supported Domain?"}
    B -->|"No"| C["OUT_OF_SCOPE"]
    B -->|"Yes"| D{"High Risk?"}
    D -->|"Yes"| E["ESCALATE"]
    D -->|"No"| F["Continue to Retrieval"]
```

The risk layer handles signals associated with sensitive support cases
such as:

-   billing,
-   fraud,
-   security,
-   account access,
-   service outages,
-   other high-risk conditions.

The design deliberately favors escalation when the system cannot safely
support an automatic response.

------------------------------------------------------------------------

# Retrieval

Retrieval is the core RAG layer.

``` mermaid
flowchart TD
    A["Normalized Query"] --> B["Retrieval Router"]

    B --> C["TF-IDF Retriever"]
    B --> D["Dense Retriever"]

    C --> E["Lexical Candidates"]
    D --> F["Semantic Candidates"]

    E --> G["Hybrid Fusion"]
    F --> G

    G --> H["Combined Candidates"]
    H --> I["Score-Based Reranking"]
    I --> J["Ranked Evidence"]
```

## TF-IDF Retrieval

The lexical retriever represents the query and knowledge chunks in
TF-IDF space and ranks candidates using cosine similarity.

This is useful for exact terminology and domain-specific phrases.

## Dense Retrieval

The semantic retriever uses:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

to create dense representations of queries and document chunks.

This is intended to improve retrieval when a ticket and relevant
documentation use different wording.

## Hybrid Retrieval

The configured hybrid score is:

``` text
Hybrid Score
=
0.30 × TF-IDF Score
+
0.70 × Dense Score
```

Thus:

``` text
TF-IDF: 30%
Dense:  70%
```

The hybrid architecture combines lexical and semantic evidence rather
than replacing one with the other.

------------------------------------------------------------------------

# Ranking and Context

## Score-Based Reranking

After initial retrieval and fusion, candidates can be reordered using
score-based reranking.

The active V2 design does **not** claim cross-encoder reranking as an
implemented feature.

A cross-encoder can be considered as a future improvement.

## Context Filtering

The context layer:

1.  removes duplicate/overlapping evidence,
2.  filters weak context,
3.  limits the final context to a maximum of three chunks.

``` text
Retrieved Candidates
        ↓
Deduplicate
        ↓
Filter
        ↓
Top 3 Chunks
        ↓
Grounding
```

------------------------------------------------------------------------

# Grounding

Grounding determines whether the retrieved evidence actually supports
the ticket.

The evaluator tracks signals including:

-   evidence availability,
-   evidence relevance,
-   token overlap.

It produces one of:

``` text
GROUNDED
PARTIAL
UNGROUNDED
```

``` mermaid
flowchart TD
    A["Filtered Evidence"] --> B{"Evidence Available?"}
    B -->|"No"| C["UNGROUNDED"]
    B -->|"Yes"| D{"Evidence Relevant?"}
    D -->|"Yes"| E["GROUNDED"]
    D -->|"Weak / Incomplete"| F["PARTIAL"]

    C --> G["Bounded Retry"]
    F --> G
    E --> H["Response Extraction"]
```

### GROUNDED

Sufficient relevant evidence is available.

### PARTIAL

Some evidence is useful, but it does not completely support the
requested answer.

### UNGROUNDED

Available evidence does not sufficiently support an answer.

The system should not fabricate missing information.

------------------------------------------------------------------------

# Deterministic Response Generation

V2 deliberately does not use an LLM API for final response generation.

Instead:

``` mermaid
flowchart LR
    A["Grounded Evidence"] --> B["Candidate Sentences"]
    B --> C["Similarity / Overlap"]
    C --> D["Actionability Filtering"]
    D --> E["Supported Sentences"]
    E --> F["Deterministic Response"]
```

Candidate sentences are selected from validated support documentation
using retrieval and overlap signals.

This provides a clear boundary:

``` text
Retrieved evidence
       ↓
Validated evidence
       ↓
Extractive response
```

rather than:

``` text
Retrieved evidence
       ↓
Unconstrained generation
       ↓
Potential unsupported claim
```

LLM-based generation is intentionally deferred to a later application
layer.

------------------------------------------------------------------------

# Reflection and Retry

The reflection stage is a bounded, Self-RAG-inspired control mechanism.

It evaluates whether evidence is sufficient and decides whether
retrieval should be attempted again.

It is **not** an autonomous agent and does not perform unlimited
self-reflection.

``` mermaid
flowchart TD
    A["Grounding Result"] --> B{"Sufficient?"}
    B -->|"Yes"| C["Continue"]
    B -->|"No"| D["Retry Policy"]
    D --> E{"Attempts < 2?"}
    E -->|"Yes"| F["Retry Retrieval"]
    F --> A
    E -->|"No"| G["ESCALATE"]
```

Configured maximum:

``` text
Maximum retries / attempts: 2
```

The bounded policy prevents infinite loops and keeps execution
predictable.

------------------------------------------------------------------------

# Final Decision

The final decision combines:

-   domain validity,
-   risk,
-   retrieval evidence,
-   grounding,
-   response availability.

Possible outcomes:

``` text
ANSWER
PARTIAL_ANSWER_ESCALATE
ESCALATE
OUT_OF_SCOPE
```

``` mermaid
flowchart TD
    A["Pipeline State"] --> B{"Out of Scope?"}
    B -->|"Yes"| C["OUT_OF_SCOPE"]
    B -->|"No"| D{"High Risk?"}
    D -->|"Yes"| E["ESCALATE"]
    D -->|"No"| F{"Grounding"}

    F -->|"GROUNDED"| G["ANSWER"]
    F -->|"PARTIAL"| H["PARTIAL_ANSWER_ESCALATE"]
    F -->|"UNGROUNDED"| E
```

A successful retrieval is therefore not automatically equivalent to
permission to answer.

------------------------------------------------------------------------

# Module Responsibilities

  Layer        Module                       Responsibility
  ------------ ---------------------------- ------------------------------------
  Ingestion    `loader.py`                  Load support documentation
  Ingestion    `chunker.py`                 Create retrieval-sized chunks
  Ingestion    `indexer.py`                 Build retrieval representations
  Query        `analyzer.py`                Analyze ticket/domain/risk signals
  Query        `rewriter.py`                Normalize and expand the query
  Query        `router.py`                  Route the query to retrieval
  Retrieval    `base.py`                    Retrieval interface
  Retrieval    `tfidf.py`                   Lexical retrieval
  Retrieval    `dense.py` / `semantic.py`   Dense semantic retrieval
  Retrieval    `hybrid.py`                  30/70 lexical-semantic fusion
  Ranking      `reranker.py`                Score-based reranking
  Context      `filter.py`                  Filter and deduplicate evidence
  Context      `compressor.py`              Compress selected evidence
  Generation   `generator.py`               Deterministic/extractive response
  Reflection   `evaluator.py`               Evaluate evidence/output
  Reflection   `retry_policy.py`            Bound retries
  Safety       `risk.py`                    Risk detection
  Safety       `grounding.py`               Grounding state
  Safety       `escalation.py`              Escalation decisions
  Pipeline     `rag_pipeline.py`            Orchestrate the full RAG flow
  Evaluation   `evaluate.py`                Evaluation
  Evaluation   `benchmark.py`               Retrieval comparison
  Evaluation   `metrics.py`                 Metrics

------------------------------------------------------------------------

# Repository Structure

The intended modular organization is:

``` text
support-triage-rag/
│
├── code/
│   ├── main.py
│   ├── config/
│   │
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── indexer.py
│   │
│   ├── query/
│   │   ├── analyzer.py
│   │   ├── rewriter.py
│   │   └── router.py
│   │
│   ├── retrieval/
│   │   ├── base.py
│   │   ├── tfidf.py
│   │   ├── dense.py
│   │   ├── semantic.py
│   │   ├── hybrid.py
│   │   └── router.py
│   │
│   ├── ranking/
│   │   └── reranker.py
│   │
│   ├── context/
│   │   ├── filter.py
│   │   └── compressor.py
│   │
│   ├── generation/
│   │   └── generator.py
│   │
│   ├── reflection/
│   │   ├── evaluator.py
│   │   └── retry_policy.py
│   │
│   ├── safety/
│   │   ├── risk.py
│   │   ├── grounding.py
│   │   └── escalation.py
│   │
│   ├── pipeline/
│   │   └── rag_pipeline.py
│   │
│   └── evaluation/
│       ├── evaluate.py
│       ├── benchmark.py
│       └── metrics.py
│
├── data/
│   ├── hackerrank/
│   ├── claude/
│   └── visa/
│
├── support_tickets/
│   ├── support_tickets.csv
│   └── baseline_output.csv
│
├── indexes/
├── results/
├── tests/
├── logs/
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

# Evaluation

The final V2 evaluation uses:

``` text
200 support tickets
```

The system is compared against the established baseline.

## Important terminology

The primary comparison metric is **Decision Parity**, not accuracy.

Decision parity measures how consistently the new pipeline reproduces
the verified baseline decisions. It is not equivalent to
supervised-learning accuracy because the evaluation does not represent a
human-labeled ground-truth classification benchmark.

------------------------------------------------------------------------

# Results

Final recorded V2 evaluation:

  Metric                             Result
  ------------------- ---------------------
  Total tickets                     **200**
  Answered                           **71**
  Escalated                         **128**
  Out of Scope                        **1**
  High Risk                          **96**
  Retrieval Retries                  **33**
  Average Attempts                 **0.68**
  Decision Parity                 **88.5%**
  Divergent Replies                  **23**
  Total Latency                 **36.78 s**
  Average Latency       **\~184 ms/ticket**

## Ticket outcomes

``` text
ANSWERED       71 / 200  (35.5%)
ESCALATED     128 / 200  (64.0%)
OUT_OF_SCOPE    1 / 200   (0.5%)
```

## Interpretation

-   **88.5% Decision Parity** indicates agreement with the established
    baseline on the evaluated decision behavior.
-   **71 answered tickets** reached an answer outcome.
-   **128 escalated tickets** were routed away from automatic answering
    because of risk, evidence, grounding, or decision conditions.
-   **1 ticket** was identified as out of scope.
-   **96 tickets** triggered high-risk conditions.
-   **33 retrieval retries** were triggered by insufficient initial
    evidence.
-   **23 divergent replies** differed from baseline reply behavior.
-   **36.78 seconds** was the total recorded evaluation latency across
    200 tickets.

------------------------------------------------------------------------

# Windows Semantic Retrieval Limitation

The dense retriever uses:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

During bulk execution on the development Windows environment, PyTorch
encountered native runtime/access-violation problems.

An explicit CPU execution path was tested. Isolated semantic retrieval
tests could run, but the bulk encoding workload remained unstable.

Therefore, the final recorded Windows evaluation used:

``` text
TF-IDF retrieval
```

while the architecture still contains:

``` text
TF-IDF
Dense Retrieval
Hybrid Retrieval
```

This limitation is documented rather than hidden. The hybrid design can
be exercised in a compatible environment.

------------------------------------------------------------------------

# Testing

Core behavior is covered by tests for areas such as:

-   pipeline execution,
-   retrieval,
-   semantic retrieval,
-   grounding,
-   decision behavior,
-   retry behavior,
-   component integration.

Run:

``` bash
python -m unittest discover -s tests
```

------------------------------------------------------------------------

# Installation and Usage

## Requirements

Python 3.9+ is recommended for the project environment.

Create a virtual environment:

``` bash
python -m venv .venv
```

### Windows

``` powershell
.venv\Scripts\activate
```

### Linux/macOS

``` bash
source .venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

## Run the pipeline

``` bash
python code/main.py
```

## Run evaluation

``` bash
python code/evaluate.py
```

Evaluation artifacts are written under:

``` text
results/
```

## Semantic retrieval dependency

``` bash
pip install sentence-transformers
```

The dense retriever uses:

``` text
all-MiniLM-L6-v2
```

The active retrieval mode depends on the configured execution
environment.

------------------------------------------------------------------------

# Design Decisions

## Why modular retrieval?

Lexical and semantic retrieval solve different retrieval problems.

``` text
TF-IDF
  ↓
Strong lexical matching

Dense Retrieval
  ↓
Semantic similarity

Hybrid
  ↓
Combined signal
```

The abstraction allows retrieval strategies to evolve independently.

## Why hybrid retrieval?

The configured score is:

``` text
Hybrid = 0.30 × TF-IDF + 0.70 × Dense
```

This gives semantic retrieval the larger contribution while retaining
lexical evidence.

## Why reranking?

Retrieval finds candidates; reranking prioritizes them.

Separating those responsibilities makes the retrieval layer easier to
modify and evaluate.

## Why only three context chunks?

The system prioritizes a small set of strong evidence rather than
passing every retrieved result downstream.

## Why grounding?

Retrieving text does not guarantee that the text answers the ticket.

Grounding makes evidence sufficiency explicit.

## Why bounded retries?

Retries provide recovery from weak retrieval without allowing an
uncontrolled loop.

## Why deterministic generation?

V2 is primarily an architectural evolution of the **retrieval/evidence
layer**. Keeping generation deterministic makes it easier to measure
retrieval, grounding, safety, and decision behavior independently of an
external LLM.

------------------------------------------------------------------------

# Explicit Non-Goals

The current repository intentionally does **not** include:

``` text
✗ External LLM API generation
✗ OpenAI / Anthropic / Gemini API calls
✗ FastAPI
✗ React / Next.js frontend
✗ Cloud deployment
✗ AWS infrastructure
✗ Conversation memory
✗ Persistent chat history
✗ Vector database
✗ GraphRAG
✗ Knowledge graph
✗ Multi-database architecture
✗ Autonomous agent loop
✗ Unbounded reflection
```

These are future application-level possibilities, not missing
requirements for the current RAG engine.

------------------------------------------------------------------------

# Relationship to the Original Project

The repositories are intentionally separate.

## Original Support Triage Agent

``` text
support-triage-agent
```

Focus:

``` text
Rule-Based NLP
      ↓
Classification
      ↓
Risk Analysis
      ↓
TF-IDF Retrieval
      ↓
Evidence Validation
      ↓
Extractive Response
      ↓
Reply / Escalate
```

## Support Triage RAG

``` text
support-triage-rag
```

Focus:

``` text
Knowledge Ingestion
      ↓
Query Processing
      ↓
Retrieval Routing
      ↓
TF-IDF / Dense / Hybrid
      ↓
Score Fusion
      ↓
Reranking
      ↓
Context Filtering
      ↓
Grounding
      ↓
Bounded Reflection
      ↓
Deterministic Response
      ↓
Safety-Aware Decision
```

The separation makes the architectural progression visible while
allowing each repository to stand independently.

------------------------------------------------------------------------

# Limitations

-   **Corpus dependence:** responses are limited to available support
    documentation.
-   **English-focused:** the current corpus and evaluation target
    English tickets.
-   **Deterministic generation:** V2 does not use an LLM API for
    response generation.
-   **Semantic runtime limitation:** bulk dense encoding was unstable on
    the development Windows environment.
-   **Baseline-based evaluation:** decision parity is not
    human-ground-truth accuracy.
-   **Limited domains:** the current knowledge base covers HackerRank,
    Claude, and Visa.
-   **Conservative escalation:** safety and grounding constraints can
    increase the escalation rate.

------------------------------------------------------------------------

# Future Direction

The next application layer can place an API, frontend, memory, and
optional LLM generation around the evidence engine:

``` mermaid
flowchart TD
    A["Frontend"] --> B["FastAPI Backend"]
    B --> C["Support Triage RAG Engine"]
    C --> D["Retrieval / Grounding / Safety"]
    D --> E["Optional LLM Generation"]
    B --> F["Conversation History / Memory"]
    C --> G["Deployment / Infrastructure"]
```

Potential future work:

-   LLM-based response generation,
-   provider abstraction,
-   FastAPI backend,
-   frontend interface,
-   conversation history,
-   persistent memory,
-   cloud/container deployment,
-   stronger human-reviewed evaluation,
-   improved semantic retrieval infrastructure,
-   learned reranking,
-   multilingual retrieval.

These are future directions and are not part of the current
implementation.

------------------------------------------------------------------------

# Research Reference

The architectural direction was informed by:

**Gao et al., "Retrieval-Augmented Generation for Large Language Models:
A Survey"**

The survey describes the evolution from:

``` text
Naive RAG
   ↓
Advanced RAG
   ↓
Modular RAG
```

This project follows the modular-RAG direction by separating:

``` text
Ingestion
Query Processing
Retrieval
Ranking
Context Processing
Grounding
Reflection
Safety
Evaluation
```

Reference:

https://arxiv.org/abs/2312.10997

This project is an independent implementation and does not claim to
reproduce the paper's methods exactly.

------------------------------------------------------------------------

# Summary

Support Triage RAG is an **evidence-first modular RAG system for
support-ticket triage**.

Its central pipeline is:

``` text
Retrieve
   ↓
Rank
   ↓
Filter
   ↓
Ground
   ↓
Reflect when necessary
   ↓
Respond only when supported
   ↓
Escalate when evidence or safety is insufficient
```

The system treats RAG as more than a retrieval component. It explicitly
controls the path from support documentation to a final triage decision.

The current implementation provides a modular foundation for later LLM
generation and application deployment while keeping retrieval,
grounding, safety, and evaluation independently observable.
