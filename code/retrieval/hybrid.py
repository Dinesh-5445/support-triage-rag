from retrieval.base import BaseRetriever

class HybridRetriever(BaseRetriever):
    def __init__(self, tfidf_retriever, dense_retriever, tfidf_weight=0.3, dense_weight=0.7):
        self.tfidf = tfidf_retriever
        self.dense = dense_retriever
        self.tfidf_weight = tfidf_weight
        self.dense_weight = dense_weight

    def retrieve(self, query, expected_domain, top_k=3):
        lexical_results = self.tfidf.retrieve(query, expected_domain, top_k=top_k*2)
        dense_results = self.dense.retrieve(query, expected_domain, top_k=top_k*2)
        
        # Hybrid Fusion
        fused_scores = {}
        chunks_by_content = {}
        
        # Add lexical scores
        for res in lexical_results:
            content = res["content"]
            fused_scores[content] = fused_scores.get(content, 0) + (res["score"] * self.tfidf_weight)
            chunks_by_content[content] = res
            
        # Add dense scores
        for res in dense_results:
            content = res["content"]
            fused_scores[content] = fused_scores.get(content, 0) + (res["score"] * self.dense_weight)
            if content not in chunks_by_content:
                chunks_by_content[content] = res
                
        # Sort and return
        sorted_contents = sorted(fused_scores.keys(), key=lambda c: fused_scores[c], reverse=True)
        
        retrieved = []
        for c in sorted_contents[:top_k]:
            chunk = chunks_by_content[c]
            chunk["score"] = fused_scores[c]  # Update score to hybrid score
            retrieved.append(chunk)
            
        return retrieved
