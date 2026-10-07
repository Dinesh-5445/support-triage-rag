import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification import classify_ticket

class QueryAnalyzer:
    def __init__(self):
        pass

    def analyze(self, ticket_text, company_norm):
        request_type, product_area, mode = classify_ticket(ticket_text, company_norm)
        return {
            "request_type": request_type,
            "product_area": product_area,
            "classification_mode": mode
        }
