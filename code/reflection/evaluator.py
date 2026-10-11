class ReflectionEvaluator:
    def __init__(self):
        pass

    def evaluate_evidence(self, context_chunks, retrieval_scores):
        if not context_chunks:
            return {
                "is_sufficient": False,
                "reason": "No context chunks retrieved."
            }
            
        top_score = context_chunks[0].get("score", 0)
        
        if top_score < 0.2:
            return {
                "is_sufficient": False,
                "reason": f"Top retrieval score ({top_score:.2f}) is too low."
            }
            
        return {
            "is_sufficient": True,
            "reason": "Evidence appears sufficient for generation."
        }
