import re
import json
import requests
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_settings

def call_openrouter(system_prompt, user_prompt):
    settings = get_settings()
    if not settings.USE_LLM:
        return None
        
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/llama-3-8b-instruct:free",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"OpenRouter API Error: {e}")
        print("Disabling LLM for remaining tickets due to API failure.")
        settings.USE_LLM = False
        return None

def classify_ticket(text, company):
    text_lower = text.lower()
    mode_used = "Rule-based"
    settings = get_settings()
    
    if settings.USE_LLM:
        system = "You are a ticket classifier. Reply ONLY with a valid JSON: {\"request_type\": \"...\", \"product_area\": \"...\"}. request_type must be one of: product_issue, feature_request, bug, invalid. product_area must be HackerRank, Claude, Visa, or Unknown."
        user = f"Classify this ticket. Company field: {company}\nTicket: {text}"
        res = call_openrouter(system, user)
        if res:
            try:
                match = re.search(r'\{.*\}', res, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    mode_used = "LLM"
                    return data.get("request_type", "product_issue"), data.get("product_area", company), mode_used
            except Exception:
                pass

    bug_keywords = ["error", "crash", "down", "not working", "failing", "blocker", "vulnerability"]
    feature_keywords = ["i want", "can you add", "suggestion", "would like", "feature"]
    invalid_keywords = ["actor in", "give me the code", "delete all files", "actor", "joke"]
    
    request_type = "product_issue"
    if any(k in text_lower for k in invalid_keywords):
        request_type = "invalid"
    elif any(k in text_lower for k in bug_keywords):
        request_type = "bug"
    elif any(k in text_lower for k in feature_keywords):
        request_type = "feature_request"
        
    product_area = company if company != "None" else "Unknown"
    
    if product_area == "Unknown":
        if any(w in text_lower for w in ["payment", "card", "refund", "visa"]):
            product_area = "Visa"
        elif any(w in text_lower for w in ["test", "candidate", "assessment", "hackerrank"]):
            product_area = "HackerRank"
        elif any(w in text_lower for w in ["conversation", "chat", "delete", "claude"]):
            product_area = "Claude"
        elif request_type != "invalid":
            product_area = "HackerRank"
            
    return request_type, product_area, mode_used

def analyze_risk(text, request_type):
    text_lower = text.lower()
    mode_used = "Rule-based"
    settings = get_settings()
    
    if settings.USE_LLM:
        system = "Analyze risk. Reply ONLY with HIGH or LOW. High risk = billing, payments, fraud, security, login access, ambiguous/very short requests."
        user = f"Ticket: {text}"
        res = call_openrouter(system, user)
        if res:
            if "HIGH" in res.upper():
                return "HIGH", "LLM"
            elif "LOW" in res.upper():
                return "LOW", "LLM"

    high_risk_keywords = [
        "refund", "payment", "billing", "charge", "money", "invoice", "spend",
        "fraud", "stolen", "identity theft", "scam", "unauthorized",
        "locked out", "lost access", "restore access", "admin removed",
        "cancel", "pause subscription",
        "vulnerability", "security", "bug bounty", "compromised",
        "score", "dispute", "increase score", "influence", "recruiter",
        "down", "outage", "stopped working", "failing", "broken",
        "infosec", "security form", "compliance", "filling in the forms",
        "urgent cash", "cash", "merchant", "seller", "not working"
    ]
    
    if any(k in text_lower for k in high_risk_keywords):
        return "HIGH", mode_used
        
    if "ignore previous" in text_lower or "system prompt" in text_lower:
        return "HIGH", mode_used
        
    if len(text.split()) < 5 and request_type != "invalid":
        return "HIGH", mode_used
        
    return "LOW", mode_used
