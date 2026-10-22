import csv
import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add code/ dir to path for sibling module resolution
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from config import BASE_DIR, DATA_DIR, LOG_FILE
from utils import preprocess_ticket, log_step
from pipeline.rag_pipeline import RAGPipeline

OUTPUT_CSV = BASE_DIR / "support_tickets" / "output.csv"

def main():
    print("Initializing Modular Advanced RAG Support Triage Agent...")
    
    # Overwrite log.txt at start of run
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        pass

    print("Loading corpus and building indices...")
    pipeline = RAGPipeline(DATA_DIR, use_semantic=False)

    tickets_file = BASE_DIR / "support_tickets" / "support_tickets.csv"
    if not tickets_file.exists():
        print(f"ERROR: Tickets file not found at {tickets_file}")
        return

    print(f"Using Data Source: {tickets_file.name}")

    tickets = []
    with open(tickets_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickets.append(row)

    results = []

    for i, ticket in enumerate(tickets):
        issue = ticket.get("Issue", "")
        subject = ticket.get("Subject", "")
        company = ticket.get("Company", "None")

        # 1: Preprocess
        ticket_text, company_norm = preprocess_ticket(issue, subject, company)

        # Process through RAG Pipeline
        result = pipeline.process(ticket_text, company_norm)
        results.append(result)

        # Structured Logging
        log_step("----- NEW TICKET (MODULAR RAG) -----")
        log_step(f"Ticket: {ticket_text}")
        log_step(f"Classification: {result['request_type']} | {result['product_area']}")
        log_step(f"Decision: {result['status']}")
        log_step(f"Justification: {result['justification']}")
        log_step(f"Confidence: {result['confidence_score']}")
        log_step(f"Latency: {result['latency']:.2f}s")
        log_step("------------------------------------")

    print(f"Writing results to {OUTPUT_CSV}...")
    fieldnames = [
        "status", "product_area", "response", "justification", 
        "request_type", "confidence_score", "failure_type", "latency",
        "retrieval_mode", "reranker_used", "grounding_status", "retrieval_attempts"
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Writing logs to {LOG_FILE}...")
    print("Done!")

if __name__ == "__main__":
    main()
