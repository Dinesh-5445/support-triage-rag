import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import STOPWORDS, LOG_FILE

def tokenize_and_filter(text):
    words = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return [w for w in words if w not in STOPWORDS]

def preprocess_ticket(issue, subject, company):
    issue = str(issue).strip() if issue else ""
    subject = str(subject).strip() if subject else ""
    company = str(company).strip() if company else "None"
    
    text = f"Subject: {subject}\nIssue: {issue}".strip()
    text = re.sub(r'\s+', ' ', text)
    return text, company

def log_step(message):
    """Appends a message to log.txt."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")
