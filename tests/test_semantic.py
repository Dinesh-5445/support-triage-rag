"""
tests/test_semantic.py — Dense Retrieval Smoke Test

Tests:
1. Model loads on CPU without errors
2. A small set of documents can be encoded
3. Embedding shape is correct and values are finite
4. A query can be encoded and similarity computed
5. Top-k results are returned with correct structure
6. Test against a real slice of the support corpus
"""
import sys
import os
import unittest

# Add code/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

DENSE_STATUS = "UNKNOWN"


class TestSemanticRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global DENSE_STATUS
        from retrieval.semantic import SEMANTIC_AVAILABLE
        if not SEMANTIC_AVAILABLE:
            DENSE_STATUS = "NOT RUN — sentence-transformers not installed"
            raise unittest.SkipTest(DENSE_STATUS)

        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            cls.np = np

            print("\n[SemanticTest] Loading model on CPU...")
            cls.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            DENSE_STATUS = "ACTIVE"
            print("[SemanticTest] Model loaded.")
        except Exception as e:
            DENSE_STATUS = f"NOT RUN — {e}"
            raise unittest.SkipTest(DENSE_STATUS)

    def test_01_encode_documents(self):
        """Model should encode a small set of docs and return correct shape."""
        docs = [
            "How do I reset my HackerRank password?",
            "My Visa card was declined at the store.",
            "Claude API is returning 529 errors.",
        ]
        embeddings = self.model.encode(docs, device="cpu")
        self.assertEqual(embeddings.shape[0], 3, "Should return 3 embeddings")
        self.assertEqual(embeddings.shape[1], 384, "MiniLM-L6-v2 dim should be 384")

    def test_02_embeddings_finite(self):
        """All embedding values must be finite (no NaN/Inf)."""
        np = self.np
        docs = ["test document"]
        emb = self.model.encode(docs, device="cpu")
        self.assertTrue(np.all(np.isfinite(emb)), "Embeddings contain non-finite values")

    def test_03_cosine_similarity(self):
        """Cosine similarity should be in [-1, 1] and semantically similar sentences score higher."""
        np = self.np
        similar_a = "I cannot log in to my account"
        similar_b = "Login is not working, I'm locked out"
        unrelated  = "The stock market went up today"

        embs = self.model.encode([similar_a, similar_b, unrelated], device="cpu")
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_norm = embs / (norms + 1e-10)

        sim_ab = float(embs_norm[0] @ embs_norm[1])
        sim_au = float(embs_norm[0] @ embs_norm[2])

        self.assertGreater(sim_ab, sim_au,
            f"Similar pair ({sim_ab:.3f}) should score higher than unrelated ({sim_au:.3f})")

    def test_04_retriever_init_and_corpus(self):
        """SemanticRetriever should init, load corpus, and index chunks."""
        from retrieval.semantic import SemanticRetriever
        from config import DATA_DIR
        r = SemanticRetriever(DATA_DIR)
        self.assertGreater(len(r.corpus_chunks), 0, "Corpus should have chunks")
        self.assertIsNotNone(r.embeddings_norm, "Normalized embeddings must exist")
        self.assertEqual(r.embeddings_norm.shape[0], len(r.corpus_chunks),
            "One embedding per chunk")

    def test_05_retrieve_returns_results(self):
        """retrieve() should return ranked results for a known query."""
        from retrieval.semantic import SemanticRetriever
        from config import DATA_DIR
        r = SemanticRetriever(DATA_DIR)
        results = r.retrieve("I cannot log in to my HackerRank account", "hackerrank", top_k=3)
        self.assertGreater(len(results), 0, "Should retrieve at least one result")
        for res in results:
            self.assertIn("content", res)
            self.assertIn("score", res)
            self.assertIn("domain", res)
            self.assertIsInstance(res["score"], float)
            self.assertGreaterEqual(res["score"], 0.0)

    def test_06_retrieve_respects_domain(self):
        """Results for a HackerRank query should come from the hackerrank domain."""
        from retrieval.semantic import SemanticRetriever
        from config import DATA_DIR
        r = SemanticRetriever(DATA_DIR)
        results = r.retrieve("test submission error", "hackerrank", top_k=3)
        for res in results:
            self.assertEqual(res["domain"], "hackerrank",
                f"Domain mismatch: expected hackerrank, got {res['domain']}")

    def test_07_ood_query_does_not_produce_results_from_wrong_domain(self):
        """OOD domain string should return no results (domain filter)."""
        from retrieval.semantic import SemanticRetriever
        from config import DATA_DIR
        r = SemanticRetriever(DATA_DIR)
        results = r.retrieve("how to delete system files", "unrelated_domain", top_k=3)
        # Domain filter should prevent returning anything when domain is unknown
        self.assertEqual(len(results), 0,
            "Should return no results for an unrecognized domain")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSemanticRetriever)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"\nDENSE RETRIEVAL STATUS: {DENSE_STATUS}")
    sys.exit(0 if result.wasSuccessful() else 1)
