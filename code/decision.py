def decide_action(risk_level, request_type, company, text):
    if risk_level == "HIGH":
        return "escalated", "High risk detected", "HIGH_RISK"
        
    if request_type == "invalid":
        if "delete" in text.lower() or "hack" in text.lower():
            return "escalated", "Potentially harmful invalid request", "HIGH_RISK"
        return "replied", "Out of scope", "NONE"
        
    if company == "Unknown" and len(text.split()) < 10:
        return "escalated", "Ambiguous intent and no domain", "LOW_RETRIEVAL_CONFIDENCE"
        
    if text.count("?") > 2:
        return "escalated", "Multiple intents detected", "MULTI_INTENT"
        
    return "proceed_to_retrieval", "Safe to process", "NONE"
