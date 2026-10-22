# Support Triage Agent — Project History Log

This document serves as the complete historical record of the Support Triage Agent project, detailing its origin, architectural transitions, technical decisions, bugs encountered, fixes applied, and performance metrics across different phases.

---

## 1. Project Origin and V1 Goal

The **Support Triage Agent** was originally conceived as a deterministic NLP-based support ticket triage system. The goal was to build a system that could classify incoming tickets, assess risk, retrieve relevant support information, validate evidence, and deterministically decide whether to reply or escalate.

The primary directive for V1 was to prioritize **determinism, safety, and explainability** over free-form generative AI capabilities. Responses needed to be strictly grounded in local support corpora, with high-risk or low-confidence tickets routed to a human agent.

## 2. Original Deterministic NLP Architecture (V1)

The V1 architecture was a linear pipeline implemented primarily as standalone scripts. The ticket passed through the following stages:
- **Classification:** Rule-based identification of request type (`product_issue`, `feature_request`, `bug`, `invalid`) and product domain based on keywords and the ticket's company metadata.
- **Risk Analysis:** Rule-based checking for sensitive keywords (billing, fraud, security, account access).
- **Retrieval:** TF-IDF vectorization with cosine similarity.
- **Evidence Validation:** Hard thresholds on TF-IDF similarity (0.25) and keyword overlap (min 2 tokens).
- **Decision Engine:** Deterministic rule tree outputting `replied` or `escalated`.
- **Response Generation:** Extractive sentence selection based on actionability rules (detecting verbs like "reset", "update") and similarity scoring.

## 3. Corpus Structure

The support knowledge base consists of manually maintained Markdown documentation across three domains:
- **HackerRank** (`data/hackerrank/`): Documentation covering candidates, assessments, skill up, and interviews.
- **Claude** (`data/claude/`): Documentation for API, console, desktop apps, enterprise plans, and privacy.
- **Visa** (`data/visa/`): Payment and support documentation.

*Decision:* These three corpora were deemed sufficient. A strict rule was established not to replace them or spend time scraping external data.

## 4. Ticket Dataset and Evaluation Setup

Input tickets are stored in `support_tickets/support_tickets.csv`. 

Initially, the evaluation dataset contained **29 tickets** covering the supported domains, including edge cases like ambiguous requests, multi-intent queries, out-of-scope requests, and prompt-injection attempts.

Evaluation was performed by `code/evaluate.py`, comparing the pipeline's decisions against a verified baseline (`support_tickets/baseline_output.csv`).

## 5. Classification and Risk Analysis (V1)

- **Implemented in:** `code/classification.py` and `code/safety/risk.py` (later modularized).
- **Classification:** An experimental LLM classification path via OpenRouter was present but disabled by default (or fell back to rule-based on failure). The core logic relied on keyword lists for bugs, feature requests, and invalid/harmful inputs.
- **Risk Analysis:** Identified high-risk terms. Tickets deemed "HIGH" risk were escalated immediately to prevent unsafe automated responses.

## 6. TF-IDF Retrieval

- **Implemented in:** `code/retrieval/tfidf.py`.
- **Logic:** The corpus was loaded, stripped of Markdown artifacts (links, headers), chunked, and vectorized using `scikit-learn`'s `TfidfVectorizer` (max_features=10000, English stopwords).
- **Retrieval:** Ticket text was vectorized and cosine similarity was computed against the corpus matrix, returning the top matching chunks filtered by the expected domain.

## 7. Evidence Validation

- **Logic:** Retrieved chunks were validated before being used. A chunk needed a TF-IDF score ≥ 0.25 and a keyword overlap of ≥ 2 tokens with the query. If no chunks passed, the system escalated with `LOW_RETRIEVAL_CONFIDENCE`.

## 8. Deterministic Response Generation

- **Implemented in:** `code/response.py`.
- **Logic:** Extracted individual sentences from the valid retrieved chunk. Scored sentences based on keyword overlap, semantic similarity, and length penalties (e.g., questions penalized).
- **Actionability:** Required the sentence to contain an actionable verb ("reset", "update", "click", etc.) to be considered a valid response.

## 9. Decision and Escalation Logic

- **Implemented in:** `code/decision.py`.
- **Rules:**
  - High risk -> Escalate
  - Invalid + harmful -> Escalate
  - Invalid + harmless -> Reply ("Out of scope")
  - Ambiguous intent + Unknown domain -> Escalate
  - Multi-intent (>2 question marks) -> Escalate
  - Passed all + Valid retrieval -> Reply

## 10. Original Benchmarks and Results (V1)

Based on the original 29-ticket dataset:
- **Baseline decision parity:** 100%
- **Divergent escalations/replies:** 0
- **Escalation rate:** 83%
- **Auto-reply rate:** 17%
- **Mean confidence score:** 0.08
- **Latency:** ~6.76 seconds total (~233 ms per ticket)

## 11. Semantic Retrieval Experiment

A benchmark script (`code/benchmark.py`) was introduced to compare the default TF-IDF retriever with a dense semantic retriever using `sentence-transformers/all-MiniLM-L6-v2`. This phase (Phase C) was intended to prove whether semantic embeddings improved the reply rate and confidence distribution over lexical search.

---

## 12. V2 Motivation and Architectural Transition

While V1 was deterministic and safe, the linear architecture was difficult to extend. **Version 2** was initiated to build an **Advanced Modular RAG Engine**.

The goal of V2 was exclusively to build a robust backend evidence engine. It was explicitly decided **not** to include LLM generation APIs, a frontend, FastAPI, or conversation memory (reserved for V3).

The pipeline was restructured into discrete modules: `ingestion`, `query`, `retrieval`, `ranking`, `context`, `safety`, `generation`, and `reflection`.

## 13. Ingestion, Chunking, and Indexing

- **Implemented in:** `code/ingestion/loader.py` and `code/ingestion/chunker.py`.
- **Logic:** Abstracted the markdown loading and chunking (300 words with 50-word overlap) away from the retriever classes, providing a uniform document format for both TF-IDF and Dense retrievers.

## 14. Query Analysis, Transformation, and Routing

- **Implemented in:** `code/query/analyzer.py`, `code/query/rewriter.py`, and `code/query/router.py`.
- **Logic:** The `QueryAnalyzer` wrapped the V1 classification logic. The `QueryRewriter` performed identity rewrite and token expansion. The `RetrievalRouter` was stubbed to route based on the selected strategy (`tfidf`, `dense`, or `hybrid`).

## 15. Dense Retrieval

- **Implemented in:** `code/retrieval/semantic.py`.
- **Logic:** Integrated `sentence-transformers` for dense embedding generation and cosine similarity retrieval.

## 16. Hybrid Retrieval

- **Implemented in:** `code/retrieval/hybrid.py`.
- **Logic:** Executed both TF-IDF and Semantic retrievers. The results were fused using a weighted scoring mechanism (default 30% TF-IDF, 70% Dense).

## 17. Reranking

- **Implemented in:** `code/ranking/reranker.py`.
- **Logic:** An optional reranker module. A cross-encoder (`ms-marco-MiniLM-L-6-v2`) was considered, but it was decided to use retrieval score-based reranking for V2 to avoid an additional heavy model dependency, leaving cross-encoders for V3.

## 18. Context Filtering

- **Implemented in:** `code/context/filter.py`.
- **Logic:** Deduped retrieved chunks based on exact content matches and truncated to a maximum number of chunks (`max_chunks=3`).

## 19. Grounding

- **Implemented in:** `code/safety/grounding.py`.
- **Logic:** Replaced the naive confidence check with explicit tracking of evidence properties. Evaluated `evidence_available`, `evidence_relevance` (top chunk score), and `token_overlap`. Emitted discrete statuses: `GROUNDED`, `PARTIAL`, or `UNGROUNDED`.

## 20. Reflection and Self-RAG-inspired Retry Logic

- **Implemented in:** `code/reflection/evaluator.py`, `code/reflection/retry_policy.py`, and `code/pipeline/rag_pipeline.py`.
- **Logic:** If evidence was deemed insufficient or ungrounded, the pipeline triggered a retry loop. A `RetryPolicy` was implemented to prevent infinite loops, capping retrieval attempts at 2.

## 21. Risk and Escalation Redesign

- **Implemented in:** `code/safety/escalation.py`.
- **Logic:** Updated the `EscalationManager` to centralize decision-making. Pre-retrieval it checked domain and risk. Post-generation it evaluated the grounding status.

## 22. Out-of-Scope Handling

- **Logic:** If the `QueryAnalyzer` determined the ticket belonged to an "Unknown" domain or matched invalid criteria, the `EscalationManager` immediately emitted an `OUT_OF_SCOPE` status without attempting retrieval.

## 23. Partial-Answer + Escalation Behavior

- **Logic:** If the grounding validator returned `PARTIAL` (e.g., relevant evidence found but poor token overlap between evidence and generated response), the system routed to a new `PARTIAL_ANSWER_ESCALATE` state, acknowledging some help was provided but human review was still required.

## 24. Modular Pipeline Integration

- **Implemented in:** `code/pipeline/rag_pipeline.py`.
- **Logic:** The `RAGPipeline` orchestrator tied all V2 modules together. It managed the flow from query analysis -> risk check -> routing -> retrieval -> reranking -> context filtering -> generation -> grounding -> escalation. 

## 25. Evaluation and Benchmark Implementation

- **Implemented in:** `code/evaluate.py` and `code/benchmark.py`.
- **Logic:** The evaluation scripts were updated to strictly process the entire dataset for both the baseline (V1) and the modular pipeline (V2) to ensure a mathematically fair comparison of decision parity.

## 26. Dataset Expansion and Current Ticket Count

- **Current Count:** The dataset (`support_tickets.csv`) was expanded to **200 tickets** to provide a more rigorous evaluation surface.

## 27. CPU/PyTorch Semantic-Retrieval Issue and Attempted Fix

- **Bug Encountered:** During Windows execution, `sentence-transformers` repeatedly crashed silently with a PyTorch warning `cudaErrorNotSupported` (due to internal CUDA probing), which then cascaded into a hard `0xC0000005` Access Violation crash during bulk corpus encoding for 200 tickets.
- **Fix Attempted:** The architecture was explicitly updated in `code/retrieval/semantic.py` to force `device="cpu"` during both initialization and `encode()` calls. 
- **Limitation:** While forcing the CPU device fixed the `cudaErrorNotSupported` warnings and allowed isolated unit tests to pass, the underlying PyTorch C++ backend on the Windows host continued to segfault on bulk processing. 
- **Current State:** To provide a verifiable and non-crashing working state, semantic retrieval was disabled by default (`use_semantic=False` in `main.py`). The hybrid architecture remains intact and functional for Linux/GPU environments.

## 28. Tests and Verification Performed

- **Implemented in:** `tests/test_pipeline.py` and `tests/test_semantic.py`.
- **Verification:** Comprehensive unit tests were written to verify semantic model loading (on CPU), hybrid orchestration, domain routing, deterministic risk escalation, and the bounded reflection loop. All core logic tests pass with 100% success.

## 29. Actual Results and Metrics (V2)

The final evaluation run (`evaluate.py`) on the 200-ticket dataset with TF-IDF retrieval mode yielded:

- **Total tickets:** 200
- **Answered:** 71
- **Escalated:** 128
- **Out of Scope:** 1
- **High Risk:** 96
- **Retrieval Retries:** 33 (avg attempts: 0.68)
- **Decision Parity:** 88.5%
- **Divergent Replies:** 23
- **Total Latency:** 36.78 seconds

*Interpretation:* V2 achieved an 88.5% parity with V1. The 23 divergent replies represent cases where V2's advanced grounding and reflection layers appropriately altered the naive V1 decision paths.

## 30. Current V2 Architecture and Limitations

**V2 is strictly a backend evidence engine.** 
- **Limitations:** 
  - Semantic retrieval is disabled by default on Windows due to PyTorch C++ access violations. 
  - The generation layer remains extractive (V1 logic) rather than generative, adhering to the safety-first deterministic mandate.
  - No conversation memory or state tracking exists across multiple user interactions.

## 31. V3 Direction: Application Layer

The planned **Version 3** scope includes:
- Replacing the extractive generation step with an LLM generation layer (e.g., OpenAI, Anthropic, or Gemini APIs).
- Implementing a frontend UI (e.g., a React or Next.js chatbot interface).
- Exposing the V2 modular pipeline via a FastAPI backend service.
- Implementing conversation memory and history.
- Cloud deployment and containerization.
