"""
RAG Pipeline — V2 Orchestrator

Flow:
  Query → Analysis → Risk/Pre-retrieval → Query Rewrite → Routing
  → Retrieval → Reranking → Context Filter → Evidence Sufficiency
  → [Retry loop, bounded] → Generation → Grounding → Decision

Supported outcomes:
  replied, partial_answer_escalate, escalated, out_of_scope

Retrieval modes: tfidf | dense | hybrid
Reranker: optional (USE_RERANKER env var)
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query.analyzer import QueryAnalyzer
from query.rewriter import QueryRewriter
from query.router import RetrievalRouter
from retrieval.tfidf import TFIDFRetriever
from retrieval.semantic import SemanticRetriever, SEMANTIC_AVAILABLE
from retrieval.hybrid import HybridRetriever
from ranking.reranker import Reranker
from context.filter import ContextFilter
from generation.generator import Generator
from reflection.evaluator import ReflectionEvaluator
from reflection.retry_policy import RetryPolicy
from safety.risk import RiskAnalyzer
from safety.grounding import GroundingValidator
from safety.escalation import EscalationManager

MAX_RETRIEVAL_ATTEMPTS = 2  # Bounded — prevents infinite retry loops


class RAGPipeline:
    def __init__(self, data_dir, use_semantic=True):
        self.analyzer = QueryAnalyzer()
        self.rewriter = QueryRewriter()
        self.risk_analyzer = RiskAnalyzer()
        self.escalation_manager = EscalationManager()

        self.tfidf = TFIDFRetriever(data_dir)

        if use_semantic and SEMANTIC_AVAILABLE:
            try:
                self.dense = SemanticRetriever(data_dir)
                self.hybrid = HybridRetriever(self.tfidf, self.dense)
                self.retrieval_router = RetrievalRouter(strategy="hybrid")
                self._retrieval_mode = "hybrid"
                print("  [RAGPipeline] Retrieval mode: HYBRID (TF-IDF + Dense)")
            except Exception as e:
                print(f"  [RAGPipeline] Dense retrieval failed to init: {e}")
                print("  [RAGPipeline] Falling back to TF-IDF only.")
                self.dense = None
                self.hybrid = None
                self.retrieval_router = RetrievalRouter(strategy="lexical")
                self._retrieval_mode = "tfidf"
        else:
            self.dense = None
            self.hybrid = None
            self.retrieval_router = RetrievalRouter(strategy="lexical")
            self._retrieval_mode = "tfidf"
            if not use_semantic:
                print("  [RAGPipeline] Retrieval mode: TF-IDF (semantic disabled by caller)")
            else:
                print("  [RAGPipeline] Retrieval mode: TF-IDF (sentence-transformers unavailable)")

        self.reranker = Reranker()  # reads USE_RERANKER env var
        self.context_filter = ContextFilter(max_chunks=3)
        self.evaluator = ReflectionEvaluator()
        self.retry_policy = RetryPolicy(max_attempts=MAX_RETRIEVAL_ATTEMPTS)
        self.generator = Generator(vectorizer=self.tfidf.vectorizer)
        self.grounding_validator = GroundingValidator()

    def process(self, original_query: str, company_norm: str) -> dict:
        start_time = time.time()

        # --- 1. Query Analysis ---
        analysis = self.analyzer.analyze(original_query, company_norm)
        request_type = analysis["request_type"]
        product_area = analysis["product_area"]

        # --- 2. Risk Analysis ---
        risk_info = self.risk_analyzer.analyze(original_query, request_type)
        risk_level = risk_info["risk_level"]

        # --- 3. Pre-retrieval Decision (deterministic) ---
        pre_decision = self.escalation_manager.evaluate_pre_retrieval(
            risk_level, request_type, company_norm, original_query
        )

        if pre_decision["action"] not in ("proceed_to_retrieval",):
            response_text = self._pre_retrieval_response(pre_decision["action"], request_type)
            return self._build_result(
                query=original_query,
                product_area=product_area,
                request_type=request_type,
                risk_level=risk_level,
                status=pre_decision["action"],
                reason=pre_decision["reason"],
                failure_type=pre_decision["failure_type"],
                confidence=0.0,
                latency=time.time() - start_time,
                response_text=response_text,
                retrieval_mode=self._retrieval_mode,
                retrieval_attempts=0,
                reranker_used=False,
                grounding_status="N/A",
            )

        # --- 4. Query Rewriting ---
        rewrite_result = self.rewriter.rewrite(original_query)
        queries_to_try = rewrite_result["rewritten_queries"]

        # --- 5. Retrieval Routing ---
        strategy = self.retrieval_router.route(analysis)
        if strategy == "hybrid" and self.hybrid:
            retriever = self.hybrid
            active_mode = "hybrid"
        elif strategy == "dense" and self.dense:
            retriever = self.dense
            active_mode = "dense"
        else:
            retriever = self.tfidf
            active_mode = "tfidf"

        # --- 6. Retrieval → Reranking → Context → Evidence → Generation → Grounding loop ---
        attempt = 0
        final_decision = None
        best_response = ""
        best_confidence = 0.0
        total_attempts = 0
        reranker_used = self.reranker.enabled
        last_grounding_status = "UNGROUNDED"

        while self.retry_policy.should_retry(attempt):
            attempt += 1
            total_attempts = attempt
            query_to_use = queries_to_try[min(attempt - 1, len(queries_to_try) - 1)]

            # Retrieval
            chunks = retriever.retrieve(query_to_use, product_area, top_k=3)

            # Reranking (optional)
            ranked_chunks = self.reranker.rerank(query_to_use, chunks)

            # Context Filtering
            filtered_chunks = self.context_filter.filter_and_compress(ranked_chunks)

            # Evidence Sufficiency (Reflection)
            evidence_eval = self.evaluator.evaluate_evidence(filtered_chunks, None)

            if not evidence_eval["is_sufficient"]:
                if attempt == self.retry_policy.max_attempts:
                    final_decision = self.escalation_manager.evaluate_post_generation(
                        {
                            "is_grounded": False,
                            "reason": "Insufficient evidence after max attempts.",
                            "grounding_status": "UNGROUNDED",
                            "evidence_available": bool(filtered_chunks),
                            "evidence_relevance": 0.0,
                            "token_overlap": 0,
                            "unsupported_claims": False,
                        },
                        max_retries_reached=True,
                    )
                continue

            # Generation (extractive — no LLM API)
            response, gen_score = self.generator.generate(
                original_query, filtered_chunks, request_type
            )

            top_retrieval_score = filtered_chunks[0].get("score", 0) if filtered_chunks else 0
            confidence = min((top_retrieval_score + gen_score) / 10.0, 1.0)

            # Grounding Validation
            grounding_eval = self.grounding_validator.validate(
                response, filtered_chunks, confidence
            )
            last_grounding_status = grounding_eval.get("grounding_status", "UNGROUNDED")

            final_decision = self.escalation_manager.evaluate_post_generation(
                grounding_eval,
                max_retries_reached=(attempt == self.retry_policy.max_attempts),
            )

            if final_decision["action"] in ("replied", "partial_answer_escalate"):
                best_response = response
                best_confidence = confidence
                break

        if not final_decision:
            final_decision = {
                "action": "escalated",
                "reason": "Max retrieval attempts exhausted with no grounded response.",
                "failure_type": "RETRIEVAL_FAILURE",
            }

        # Final response text
        if final_decision["action"] == "escalated":
            best_response = (
                "This issue requires further review by our support team. "
                "Please wait while we connect you to a human agent."
            )
        elif final_decision["action"] == "partial_answer_escalate":
            best_response = (
                f"Based on available support information: {best_response} "
                "However, this may not fully resolve your issue — escalating for human review."
            )

        return self._build_result(
            query=original_query,
            product_area=product_area,
            request_type=request_type,
            risk_level=risk_level,
            status=final_decision["action"],
            reason=final_decision["reason"],
            failure_type=final_decision["failure_type"],
            confidence=round(best_confidence, 2),
            latency=time.time() - start_time,
            response_text=best_response,
            retrieval_mode=active_mode,
            retrieval_attempts=total_attempts,
            reranker_used=reranker_used,
            grounding_status=last_grounding_status,
        )

    def _pre_retrieval_response(self, action: str, request_type: str) -> str:
        if action == "out_of_scope":
            return (
                "This request does not match any of the supported domains "
                "(HackerRank, Claude, Visa). It cannot be processed by this system."
            )
        if action == "escalated":
            return (
                "This issue has been flagged for immediate escalation. "
                "A support agent will review your case."
            )
        return ""

    def _build_result(self, query, product_area, request_type, risk_level,
                      status, reason, failure_type, confidence, latency,
                      response_text, retrieval_mode, retrieval_attempts,
                      reranker_used, grounding_status) -> dict:
        justification = (
            f"Domain: {product_area} | Risk: {risk_level} | "
            f"retrieval_mode: {retrieval_mode} | attempts: {retrieval_attempts} | "
            f"reranker: {'yes' if reranker_used else 'no'} | "
            f"grounding: {grounding_status} | "
            f"Decision: {status} | Reason: {reason}"
        )
        return {
            "status": status,
            "product_area": product_area,
            "response": response_text,
            "justification": justification,
            "request_type": request_type,
            "confidence_score": confidence,
            "failure_type": failure_type,
            "latency": latency,
            "retrieval_mode": retrieval_mode,
            "retrieval_attempts": retrieval_attempts,
            "reranker_used": reranker_used,
            "grounding_status": grounding_status,
        }
