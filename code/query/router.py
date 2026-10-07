class RetrievalRouter:
    def __init__(self, strategy="hybrid"):
        # Options: "lexical", "dense", "hybrid"
        self.strategy = strategy

    def route(self, query_analysis):
        # We can implement rules here based on risk or domain.
        # For this project, we'll route based on initialization strategy.
        return self.strategy
