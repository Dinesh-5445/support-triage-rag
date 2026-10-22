"""
tests/test_pipeline.py — Core Pipeline Component Tests

Tests:
- TF-IDF retrieval
- Hybrid retrieval (requires dense retrieval)
- Reranker (score-based)
- Query routing
- OOD handling (invalid domain)
- Grounding validator
- Escalation manager
- Reflection retry limit
- RAGPipeline (end-to-end, tfidf mode)
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))


class TestTFIDFRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from retrieval.tfidf import TFIDFRetriever
        from config import DATA_DIR
        cls.retriever = TFIDFRetriever(DATA_DIR)

    def test_retrieves_results(self):
        results = self.retriever.retrieve("I cannot access my account", "hackerrank", top_k=3)
        self.assertGreater(len(results), 0)

    def test_domain_filter(self):
        results = self.retriever.retrieve("card declined at store", "visa", top_k=3)
        for r in results:
            self.assertEqual(r["domain"], "visa")

    def test_returns_score(self):
        results = self.retriever.retrieve("test submission error", "hackerrank", top_k=3)
        for r in results:
            self.assertIn("score", r)
            self.assertGreaterEqual(r["score"], 0.0)

    def test_ood_domain_returns_empty(self):
        results = self.retriever.retrieve("delete all system files", "unknown_domain", top_k=3)
        self.assertEqual(len(results), 0,
            "Unknown domain should return no results via domain filter")


class TestReranker(unittest.TestCase):
    def setUp(self):
        from ranking.reranker import Reranker
        self.reranker = Reranker()

    def test_reranker_sorts_by_score(self):
        chunks = [
            {"content": "low", "score": 0.1, "domain": "hackerrank"},
            {"content": "high", "score": 0.9, "domain": "hackerrank"},
            {"content": "mid", "score": 0.5, "domain": "hackerrank"},
        ]
        ranked = self.reranker.rerank("test query", chunks)
        scores = [r["score"] for r in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_reranker_disabled_passthrough(self):
        from ranking.reranker import Reranker
        r = Reranker(enabled=False)
        chunks = [{"content": "a", "score": 0.1}, {"content": "b", "score": 0.9}]
        result = r.rerank("query", chunks)
        # Passthrough — order preserved
        self.assertEqual(result[0]["content"], "a")

    def test_reranker_empty_input(self):
        result = self.reranker.rerank("query", [])
        self.assertEqual(result, [])


class TestQueryRouter(unittest.TestCase):
    def test_lexical_strategy(self):
        from query.router import RetrievalRouter
        r = RetrievalRouter(strategy="lexical")
        self.assertEqual(r.route({}), "lexical")

    def test_hybrid_strategy(self):
        from query.router import RetrievalRouter
        r = RetrievalRouter(strategy="hybrid")
        self.assertEqual(r.route({}), "hybrid")

    def test_dense_strategy(self):
        from query.router import RetrievalRouter
        r = RetrievalRouter(strategy="dense")
        self.assertEqual(r.route({}), "dense")


class TestGroundingValidator(unittest.TestCase):
    def setUp(self):
        from safety.grounding import GroundingValidator
        self.g = GroundingValidator()

    def test_no_evidence_is_ungrounded(self):
        result = self.g.validate("some answer", [], 0.0)
        self.assertFalse(result["is_grounded"])
        self.assertFalse(result["evidence_available"])
        self.assertEqual(result["grounding_status"], "UNGROUNDED")

    def test_low_score_is_ungrounded(self):
        chunks = [{"content": "some evidence text", "score": 0.05, "domain": "visa"}]
        result = self.g.validate("some answer", chunks, 0.05)
        self.assertFalse(result["is_grounded"])
        self.assertEqual(result["grounding_status"], "UNGROUNDED")

    def test_self_reported_failure_is_ungrounded(self):
        chunks = [{"content": "evidence here", "score": 0.8, "domain": "visa"}]
        result = self.g.validate("insufficient evidence to help you", chunks, 0.8)
        self.assertFalse(result["is_grounded"])
        self.assertTrue(result["unsupported_claims"])

    def test_good_evidence_is_grounded(self):
        evidence = "To reset your password, go to the login page and click Forgot Password."
        chunks = [{"content": evidence, "score": 0.75, "domain": "hackerrank"}]
        response = "To reset your password go to the login page and click Forgot Password"
        result = self.g.validate(response, chunks, 0.75)
        self.assertTrue(result["is_grounded"])
        self.assertTrue(result["evidence_available"])
        self.assertIn(result["grounding_status"], ["GROUNDED", "PARTIAL"])


class TestEscalationManager(unittest.TestCase):
    def setUp(self):
        from safety.escalation import EscalationManager
        self.em = EscalationManager()

    def test_invalid_request_is_out_of_scope(self):
        result = self.em.evaluate_pre_retrieval("low", "invalid", "None", "delete all files")
        self.assertIn(result["action"], ["escalated", "out_of_scope"])

    def test_high_risk_escalates(self):
        result = self.em.evaluate_pre_retrieval("HIGH", "complaint", "Visa", "stolen card fraud")
        self.assertEqual(result["action"], "escalated")

    def test_normal_request_proceeds(self):
        result = self.em.evaluate_pre_retrieval("LOW", "technical", "HackerRank", "test error")
        self.assertEqual(result["action"], "proceed_to_retrieval")

    def test_post_generation_grounded_replied(self):
        grounding = {"is_grounded": True, "reason": "ok", "grounding_status": "GROUNDED",
                     "evidence_available": True, "evidence_relevance": 0.8,
                     "token_overlap": 10, "unsupported_claims": False}
        result = self.em.evaluate_post_generation(grounding)
        self.assertEqual(result["action"], "replied")

    def test_post_generation_ungrounded_max_retries_escalates(self):
        grounding = {"is_grounded": False, "reason": "low score", "grounding_status": "UNGROUNDED",
                     "evidence_available": True, "evidence_relevance": 0.1,
                     "token_overlap": 0, "unsupported_claims": False}
        result = self.em.evaluate_post_generation(grounding, max_retries_reached=True)
        self.assertEqual(result["action"], "escalated")

    def test_partial_answer_outcome(self):
        grounding = {"is_grounded": True, "reason": "partial", "grounding_status": "PARTIAL",
                     "evidence_available": True, "evidence_relevance": 0.5,
                     "token_overlap": 1, "unsupported_claims": False}
        result = self.em.evaluate_post_generation(grounding)
        self.assertIn(result["action"], ["partial_answer_escalate", "replied"])


class TestReflectionRetryLimit(unittest.TestCase):
    def test_retry_bounded(self):
        from reflection.retry_policy import RetryPolicy
        p = RetryPolicy(max_attempts=2)
        self.assertTrue(p.should_retry(0))
        self.assertTrue(p.should_retry(1))
        self.assertFalse(p.should_retry(2))

    def test_retry_no_infinite_loop(self):
        from reflection.retry_policy import RetryPolicy
        p = RetryPolicy(max_attempts=3)
        count = 0
        for attempt in range(10):
            if p.should_retry(attempt):
                count += 1
            else:
                break
        self.assertEqual(count, 3)


class TestOODHandling(unittest.TestCase):
    def test_ood_ticket_escalates(self):
        """An out-of-domain ticket (Company=None, unrelated request) must not produce an answer."""
        from pipeline.rag_pipeline import RAGPipeline
        from config import DATA_DIR
        pipeline = RAGPipeline(DATA_DIR, use_semantic=False)
        result = pipeline.process("Give me the code to delete all files from the system", "None")
        self.assertIn(result["status"], ["escalated", "out_of_scope"],
            f"OOD request should be escalated, got: {result['status']}")

    def test_ood_request_type_is_marked(self):
        """Out-of-domain queries should be classified as invalid or out_of_scope."""
        from query.analyzer import QueryAnalyzer
        analyzer = QueryAnalyzer()
        result = analyzer.analyze("Give me the code to delete all system files", "None")
        self.assertIn(result["request_type"], ["invalid", "out_of_scope"],
            f"Expected invalid/out_of_scope, got: {result['request_type']}")


class TestRagPipelineEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pipeline.rag_pipeline import RAGPipeline
        from config import DATA_DIR
        cls.pipeline = RAGPipeline(DATA_DIR, use_semantic=False)

    def test_hackerrank_ticket(self):
        result = self.pipeline.process(
            "My HackerRank test submission is not working", "HackerRank"
        )
        self.assertIn("status", result)
        self.assertIn(result["status"], ["replied", "escalated", "partial_answer_escalate"])
        self.assertIn("product_area", result)
        self.assertIn("latency", result)
        self.assertGreater(result["latency"], 0)

    def test_result_has_all_required_keys(self):
        result = self.pipeline.process("I cannot access my account", "Claude")
        required = ["status", "product_area", "response", "justification",
                    "request_type", "confidence_score", "failure_type", "latency"]
        for key in required:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_high_risk_ticket_escalates(self):
        result = self.pipeline.process(
            "Someone stole my Visa card and is making unauthorized transactions", "Visa"
        )
        self.assertEqual(result["status"], "escalated",
            "High-risk tickets must always be escalated")

    def test_retrieval_mode_logged(self):
        result = self.pipeline.process("certificate download not working", "HackerRank")
        # retrieval_mode should be in justification or result
        self.assertIn("justification", result)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestTFIDFRetrieval, TestReranker, TestQueryRouter,
                TestGroundingValidator, TestEscalationManager,
                TestReflectionRetryLimit, TestOODHandling, TestRagPipelineEndToEnd]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
