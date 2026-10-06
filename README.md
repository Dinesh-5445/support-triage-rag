# Support Triage Agent

A deterministic NLP-based support ticket triage system that classifies incoming tickets, assesses risk, retrieves relevant support information using TF-IDF and cosine similarity, validates the retrieved evidence, and decides whether to reply or escalate.

The system prioritizes **determinism, safety, and explainability** over free-form generation. Responses are grounded in a local support corpus, while high-risk or low-confidence cases are escalated.

## Overview

The project addresses multi-domain support triage across **HackerRank, Claude, and Visa**. Each ticket passes through a sequence of classification, risk analysis, retrieval, validation, decision, and response-generation stages.

The current implementation is a standalone terminal-based pipeline and does not require a deployed web application or external AI service for its core deterministic workflow.

## Key Features

- **Rule-based NLP classification** of request type and product domain
- **Deterministic risk analysis** for billing, security, account access, outages, and other sensitive cases
- **TF-IDF information retrieval** with cosine similarity over a local support corpus
- **Evidence validation** using retrieval score and keyword-overlap thresholds
- **Deterministic extractive response generation** from validated support content
- **Risk-aware decision engine** that chooses between replying and escalating
- **Explainable outputs and evaluation** including retrieval scores, confidence, decision, and failure reason
- **TF-IDF vs. semantic retrieval benchmarking** using `sentence-transformers`

## Architecture

```text
Input Ticket
     │
     ▼
Classification
     │
     ▼
Risk Analysis
     │
     ▼
Decision Engine
     │
     ├──────────────► Escalate
     │
     ▼
TF-IDF Retrieval
     │
     ▼
Evidence Validation
     │
     ├──────────────► Escalate
     │
     ▼
Response Extraction
     │
     ▼
Reply
     │
     ▼
Logging & Evaluation
```

### Pipeline Stages

| Stage | Description |
|---|---|
| **Classification** | Determines the request type and product domain using deterministic rules and ticket metadata. |
| **Risk Analysis** | Identifies high-risk cases involving areas such as billing, fraud, security, account access, or outages. |
| **Retrieval** | Searches the local support corpus using TF-IDF vectors and cosine similarity. |
| **Validation** | Rejects retrieval candidates that do not meet the required relevance and keyword-overlap conditions. |
| **Decision** | Combines risk, classification, and retrieval outcomes to determine whether to reply or escalate. |
| **Response** | Selects relevant actionable sentences from validated documentation rather than generating unsupported content. |
| **Evaluation** | Compares decisions against a verified baseline and records performance and latency metrics. |

## Dataset

Input tickets are stored in:

```text
support_tickets/support_tickets.csv
```

Each ticket contains:

- `Issue` — main ticket description
- `Subject` — ticket subject
- `Company` — associated domain such as HackerRank, Claude, or Visa

The evaluation dataset contains **29 tickets** covering the supported domains and includes edge cases such as ambiguous requests, multi-intent queries, out-of-scope requests, and prompt-injection attempts.

The support corpus is stored under:

```text
data/
├── hackerrank/
├── claude/
└── visa/
```

The corpus consists of local Markdown support documentation used as the retrieval source.

## Methodology

### 1. Classification

The default classifier is deterministic and rule-based. It identifies the request type from predefined linguistic patterns and determines the product domain using the ticket's company field and content.

The current implementation does **not** train a supervised classification model.

### 2. Risk Analysis

A deterministic risk scanner checks the ticket for sensitive or high-risk patterns, including financial, security, account-access, and service-outage related terms.

High-risk or otherwise unsafe cases are routed toward escalation rather than automatic response.

### 3. TF-IDF Retrieval

The local support corpus is converted into TF-IDF vectors using `scikit-learn`. Ticket text is transformed into the same vector space and compared with corpus chunks using cosine similarity.

The highest-ranking relevant chunks are selected while respecting the classified product domain.

### 4. Evidence Validation

Retrieved content must pass relevance checks before it can be used for a response.

The current validation combines:

- TF-IDF cosine similarity
- Keyword overlap between the ticket and retrieved content

This prevents weak or unrelated retrieval results from being used as answers.

### 5. Decision Engine

The decision engine applies deterministic rules before and after retrieval.

High-risk, ambiguous, unsupported, or low-confidence cases are escalated. Only tickets with sufficient evidence and a valid response candidate are automatically replied to.

### 6. Response Generation

Responses are generated using extractive sentence selection from validated support documentation.

Candidate sentences are scored using retrieval similarity, keyword overlap, and actionability criteria. This keeps the response grounded in the local corpus and avoids unsupported free-form claims.

## Evaluation & Benchmarking

### Evaluation

`code/evaluate.py` runs the pipeline and compares the resulting decisions against the verified baseline.

The evaluation tracks:

- Baseline decision parity
- Escalation rate
- Reply rate
- Divergent decisions
- Processing latency
- Confidence distribution

**Baseline parity is not supervised-learning accuracy.** It measures how consistently the current pipeline reproduces the verified baseline decisions.

### Benchmarking

`code/benchmark.py` compares the default TF-IDF retriever with a dense semantic retriever based on:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The benchmark compares retrieval latency, reply/escalation rates, confidence distributions, and per-ticket agreement between retrieval approaches.

## Results

The latest recorded evaluation covers **29 tickets**.

| Metric | Result |
|---|---:|
| Tickets evaluated | 29 |
| **Baseline decision parity** | **100%** |
| Divergent escalations | 0 |
| Divergent replies | 0 |
| Escalation rate | 83% |
| Auto-reply rate | 17% |
| Mean confidence score | 0.08 |
| Maximum confidence score | 0.77 |
| Total pipeline latency | ~6.76 seconds |
| **Average latency per ticket** | **~233 ms** |

### Interpretation

- **100% baseline decision parity** means all 29 tickets matched the verified baseline on the evaluated decision fields.
- **83% escalation rate** reflects the system's deliberate safety bias toward escalation when a ticket is high-risk, ambiguous, unsupported, or lacks sufficient retrieval evidence.
- **17% auto-reply rate** corresponds to 5 of the 29 evaluated tickets receiving a direct response.
- Confidence scores are internal normalized scores derived from retrieval and response-selection signals; they are **not classification probabilities**.

## Example

A low-risk ticket can follow this path:

```text
Input
  ↓
Classification → Claude / feature_request
  ↓
Risk Analysis → LOW
  ↓
TF-IDF Retrieval → relevant support chunk
  ↓
Validation → sufficient similarity + keyword overlap
  ↓
Response Extraction → actionable sentence selected
  ↓
Decision → replied
```

Example response:

```text
status: replied
product_area: Claude
response: [response extracted from validated support documentation]
failure_type: NONE
```

## Project Structure

```text
support-triage-agent/
│
├── code/
│   ├── main.py
│   ├── classification.py
│   ├── decision.py
│   ├── response.py
│   ├── config.py
│   ├── utils.py
│   ├── evaluate.py
│   ├── benchmark.py
│   └── retrieval/
│       ├── base.py
│       ├── tfidf.py
│       └── semantic.py
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
├── results/
│   ├── evaluation_results.csv
│   ├── metrics.json
│   └── log.txt
│
├── requirements.txt
└── README.md
```

## How to Run

### Prerequisites

- Python 3.9+
- Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

### Run the Triage Pipeline

```bash
python code/main.py
```

The pipeline reads the support ticket dataset and writes its output and logs to `results/`.

### Run Evaluation

```bash
python code/evaluate.py
```

Evaluation metrics are written to:

```text
results/metrics.json
```

### Run the Retrieval Benchmark

The benchmark requires the semantic retrieval dependency:

```bash
pip install sentence-transformers
python code/benchmark.py
```

Benchmark outputs are stored in `results/`.

## Limitations

- **Keyword dependence:** Rule-based classification can be sensitive to wording and paraphrases.
- **High escalation rate:** The system intentionally favors safety and escalation over aggressive automation.
- **Corpus dependence:** Responses are limited to information available in the local support corpus.
- **English-focused:** The current corpus, rules, and evaluation are designed for English-language tickets.
- **No trained classifier:** The current system uses deterministic rules and TF-IDF information retrieval rather than a trained supervised ML model.

## Future Improvements

- Train and evaluate a supervised support-ticket classifier on a larger labeled dataset.
- Combine lexical TF-IDF retrieval with semantic retrieval in a hybrid retriever.
- Expand the evaluation dataset and introduce human-reviewed ground truth.
- Add multilingual classification and retrieval.
- Explore learned risk classification while preserving the existing safety gates.
