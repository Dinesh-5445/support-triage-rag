"""
Semantic Retrieval Engine — V2
Uses sentence-transformers (all-MiniLM-L6-v2) for dense embedding retrieval.
Explicitly uses CPU device to avoid PyTorch CUDA probe issues on Windows.
This module is INDEPENDENT of TF-IDF. It does NOT replace it.
"""
import sys
import os

# Add code/ parent dir to path for sibling module resolution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

from retrieval.base import BaseRetriever
from retrieval.tfidf import load_corpus


class SemanticRetriever(BaseRetriever):
    """
    Dense embedding retriever using sentence-transformers.
    Always initializes on CPU to avoid CUDA environment probing.
    Encodes all corpus chunks at init time, then retrieves via cosine similarity.
    """
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, data_dir):
        if not SEMANTIC_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. Run: pip install sentence-transformers"
            )

        print(f"  [SemanticRetriever] Loading model: {self.MODEL_NAME} (device=cpu)")
        # Explicitly force CPU — prevents PyTorch from probing CUDA on Windows
        self.model = SentenceTransformer(self.MODEL_NAME, device="cpu")

        print("  [SemanticRetriever] Loading and encoding corpus...")
        self.corpus_chunks = load_corpus(data_dir)
        texts = [c["content"] for c in self.corpus_chunks]

        if texts:
            self.embeddings = self.model.encode(
                texts, batch_size=64, show_progress_bar=False, device="cpu"
            )
            # Normalize for cosine similarity via dot product
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings_norm = self.embeddings / (norms + 1e-10)
        else:
            self.embeddings = None
            self.embeddings_norm = None

        print(f"  [SemanticRetriever] Indexed {len(self.corpus_chunks)} chunks.")

    def retrieve(self, query: str, expected_domain: str, top_k: int = 3):
        if self.embeddings_norm is None:
            return []

        query_emb = self.model.encode([query], show_progress_bar=False, device="cpu")
        query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)

        # Cosine similarity via dot product (both normalized)
        scores = (self.embeddings_norm @ query_norm.T).flatten()

        top_indices = scores.argsort()[::-1][:top_k * 3]

        retrieved = []
        expected_domain_lower = (
            expected_domain.lower()
            if expected_domain and expected_domain not in ("Unknown", "None")
            else None
        )

        for idx in top_indices:
            chunk = self.corpus_chunks[idx]
            score = float(scores[idx])

            if expected_domain_lower and chunk["domain"] != expected_domain_lower:
                continue

            retrieved.append({
                "content": chunk["content"],
                "score": score,
                "domain": chunk["domain"]
            })

            if len(retrieved) >= top_k:
                break

        return retrieved


def validate_chunk_semantic(query, chunk, tokenize_fn, threshold=0.30, overlap_min=2):
    """
    Validation gate for semantic chunks — uses cosine score threshold + keyword overlap.
    Threshold is slightly higher than TF-IDF (0.25) because embedding scores are denser.
    """
    chunk_score = chunk["score"]
    query_tokens = set(tokenize_fn(query))
    chunk_tokens = set(tokenize_fn(chunk["content"]))
    overlap = len(query_tokens.intersection(chunk_tokens))

    if chunk_score < threshold:
        return False, f"Semantic score ({chunk_score:.2f}) < threshold ({threshold})", chunk_score, overlap

    if overlap < overlap_min:
        return False, f"keyword overlap ({overlap}) < threshold ({overlap_min})", chunk_score, overlap

    return True, "Valid semantic retrieval", chunk_score, overlap
