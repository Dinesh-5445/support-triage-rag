import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from response import construct_response_from_chunk

class Generator:
    def __init__(self, vectorizer=None):
        # Using the deterministic extractive generator for safety and baseline parity
        self.vectorizer = vectorizer

    def generate(self, original_query, context_chunks, request_type):
        if not self.vectorizer:
            return "No vectorizer provided for extractive generation.", 0.0

        best_response = ""
        best_score = 0.0
        
        # We try to extract a response from the top chunk
        for chunk in context_chunks:
            resp, _, scores = construct_response_from_chunk(
                original_query, chunk["content"], request_type, self.vectorizer
            )
            
            if resp:
                top_score = scores[0]["score"] if scores else 0.0
                if top_score > best_score:
                    best_score = top_score
                    best_response = resp
                    
        if best_response:
            return best_response, best_score
            
        return "Insufficient evidence to construct a helpful response.", 0.0
