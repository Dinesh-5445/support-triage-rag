"""
Reranker — V2
Optional cross-encoder reranking. When disabled, acts as a pass-through.
Configure via: Reranker(enabled=True/False)
Environment variable: USE_RERANKER=true/false
"""
import os


class Reranker:
    """
    Optional reranker module. When enabled=True, reranks candidates by score.
    In V2, the cross-encoder (sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2)
    is reserved for V3 since it requires a separate model download.
    The current implementation uses the retrieval score for ranking.

    Future: Replace _score_based_rerank() with a cross-encoder call.
    """

    def __init__(self, mode="score", enabled=None):
        """
        Args:
            mode: "score" (default) — rerank by retrieval score.
            enabled: bool. If None, reads USE_RERANKER env var. Defaults to True.
        """
        if enabled is None:
            env_val = os.environ.get("USE_RERANKER", "true").lower()
            self.enabled = env_val in ("true", "1", "yes")
        else:
            self.enabled = enabled
        self.mode = mode

    def rerank(self, query: str, chunks: list) -> list:
        """
        Rerank chunks. If disabled, returns chunks unchanged (pass-through).
        """
        if not self.enabled or not chunks:
            return chunks

        if self.mode == "score":
            return self._score_based_rerank(chunks)

        return chunks

    def _score_based_rerank(self, chunks: list) -> list:
        """Sort by retrieval score descending."""
        return sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
