import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import tokenize_and_filter

class QueryRewriter:
    def __init__(self):
        pass

    def rewrite(self, original_query):
        # Identity rewrite + expansion (simple extraction of meaningful tokens)
        query_tokens = tokenize_and_filter(original_query)
        expanded_query = " ".join(query_tokens)
        return {
            "original_query": original_query,
            "rewritten_queries": [original_query, expanded_query]
        }
