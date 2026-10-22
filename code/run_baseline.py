import csv
import sys
import os

# Add code/ dir to path for sibling module resolution
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from config import BASE_DIR, DATA_DIR, LOG_FILE, get_settings
from utils import preprocess_ticket, tokenize_and_filter, log_step
from classification import classify_ticket, analyze_risk
from decision import decide_action
from retrieval.tfidf import TFIDFRetriever, validate_chunk, TfidfVectorizer
from response import construct_response_from_chunk

BASELINE_OUTPUT_CSV = BASE_DIR / "support_tickets" / "baseline_output.csv"


def main():
    print("Initializing V1 Baseline — TF-IDF Support Triage Agent...") 
    
    # Overwrite log.txt at start of run
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        pass

    if not TfidfVectorizer:
        print("ERROR: Missing dependencies. Please run: pip install -r code/requirements.txt")
        return

    print("Loading corpus and building TF-IDF index...")
    retriever = TFIDFRetriever(DATA_DIR)

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
        query_tokens = tokenize_and_filter(ticket_text)

        # 2: Classify
        request_type, product_area, class_mode = classify_ticket(ticket_text, company_norm)

        # 3: Risk Analysis
        risk_level, risk_mode = analyze_risk(ticket_text, request_type)

        # 4: Decision Engine
        base_action, base_decision_reason, base_failure_type = decide_action(risk_level, request_type, company_norm, ticket_text)

        # 5: Retrieval & 6: Validation & 7: Response Construction
        retrieved_chunks = []
        chunk_score = 0.0
        chunk_overlap = 0

        action = base_action
        response = ""
        sentence_scores = []
        sentence_validity = "FALSE"
        failure_type = base_failure_type

        if action == "proceed_to_retrieval" or (action == "replied" and request_type == "invalid"):
            if request_type == "invalid":
                response = "This query is unrelated to supported domains (HackerRank, Claude, Visa), so it cannot be processed."
                action = "replied"
                sentence_validity = "N/A"
            else:
                expanded_queries = [ticket_text, " ".join(query_tokens)]
                best_chunks = []
                for q in expanded_queries:
                    if not q.strip():
                        continue
                    chunks = retriever.retrieve(q, product_area, top_k=3)
                    best_chunks.extend(chunks)

                best_chunks.sort(key=lambda x: x["score"], reverse=True)
                seen = set()
                for c in best_chunks:
                    if c["content"] not in seen:
                        retrieved_chunks.append(c)
                        seen.add(c["content"])

                extraction_success = False
                for chunk in retrieved_chunks:
                    is_valid, v_reason, c_score, c_overlap = validate_chunk(ticket_text, chunk)
                    if not is_valid:
                        continue

                    resp, ext_reason, scores = construct_response_from_chunk(
                        ticket_text, chunk["content"], request_type, retriever.vectorizer
                    )

                    if resp:
                        response = resp
                        action = "replied"
                        chunk_score = c_score
                        chunk_overlap = c_overlap
                        sentence_scores = scores
                        extraction_success = True
                        sentence_validity = "TRUE"
                        failure_type = "NONE"
                        break

                if not extraction_success:
                    action = "escalated"
                    failure_type = "LOW_RETRIEVAL_CONFIDENCE"

        if action == "escalated":
            response = "This issue requires further review by our support team. Please wait while we connect you to a human agent."
            if failure_type == "NONE":
                failure_type = "LOW_RETRIEVAL_CONFIDENCE"
            if chunk_score == 0 and chunk_overlap == 0 and company_norm in ["None", "Unknown"]:
                product_area = "Unknown"

        # Confidence Score Calculation
        confidence_score = 0.0
        if action == "replied" and request_type != "invalid":
            top_sentence_score = sentence_scores[0]["score"] if sentence_scores else 0.0
            raw_score = chunk_score + (chunk_overlap * 0.1) + top_sentence_score
            confidence_score = round(min(raw_score / 15.0, 1.0), 2)

        # Structured Decision Explanation
        justification = (
            f"Domain: {product_area} | Risk: {risk_level} | "
            f"TF-IDF Score: {chunk_score:.2f} | Keyword Overlap: {chunk_overlap} | "
            f"Sentence Validity: {sentence_validity} | Final Decision: {action}"
        )

        result = {
            "status": action,
            "product_area": product_area,
            "response": response,
            "justification": justification,
            "request_type": request_type,
            "confidence_score": confidence_score,
            "failure_type": failure_type
        }
        results.append(result)

        # 9: Strict Logging
        log_step("----- BASELINE TICKET -----")
        log_step(f"Ticket: {ticket_text}")
        log_step(f"Preprocessed: {' '.join(query_tokens)}")
        log_step(f"Classification: {request_type} | {product_area}")
        log_step(f"Risk: {risk_level} | {base_decision_reason}")
        log_step(f"Decision: {action}")
        log_step(f"TF-IDF Score: {chunk_score:.2f}")
        log_step(f"Keyword Overlap: {chunk_overlap}")
        log_step(f"Sentence Score: {sentence_scores[0]['score'] if sentence_scores else 0.0}")
        log_step(f"Selected Response: {response}")
        log_step(f"Confidence: {confidence_score}")
        log_step(f"Final Action: {action}")
        log_step("---------------------------")

    print(f"Writing baseline results to {BASELINE_OUTPUT_CSV}...")
    fieldnames = ["status", "product_area", "response", "justification", "request_type", "confidence_score", "failure_type"]

    with open(BASELINE_OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Writing logs to {LOG_FILE}...")
    print(f"Baseline complete: {len(results)} tickets processed.")


if __name__ == "__main__":
    main()
