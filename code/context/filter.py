class ContextFilter:
    def __init__(self, max_chunks=3):
        self.max_chunks = max_chunks

    def filter_and_compress(self, retrieved_chunks):
        deduped = []
        seen = set()
        
        for chunk in retrieved_chunks:
            # Simple deduplication by exact content match
            if chunk["content"] not in seen:
                deduped.append(chunk)
                seen.add(chunk["content"])
                
        # We can implement relevance filtering threshold here if needed
        # but retriever already sorts and scores.
        
        # Sort by score just in case, though they should be sorted already
        deduped.sort(key=lambda x: x["score"], reverse=True)
        
        return deduped[:self.max_chunks]
