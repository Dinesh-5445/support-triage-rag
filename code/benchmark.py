"""
benchmark.py — Phase C Benchmarking
Runs TF-IDF and Semantic retrievers on all tickets and compares:
- Reply rate, escalation rate, retrieval latency, confidence distribution
- Outputs benchmark_tfidf.json and benchmark_semantic.json
- Outputs benchmark_comparison.csv for side-by-side ticket-level comparison
Does NOT modify main.py or production output.csv.
"""
import csv
import json
import time
import sys
import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from config import BASE_DIR, DATA_DIR
from utils import preprocess_ticket, tokenize_and_filter
from classification import classify_ticket, analyze_risk
from decision import decide_action
from retrieval.tfidf import TFIDFRetriever, validate_chunk
from retrieval.semantic import SemanticRetriever, validate_chunk_semantic, SEMANTIC_AVAILABLE
from response import construct_response_from_chunk

TICKETS_FILE = BASE_DIR / "support_tickets" / "support_tickets.csv"
BENCHMARK_DIR = BASE_DIR / "results"
BENCHMARK_DIR.mkdir(exist_ok=True)


def run_retrieval_pipeline(retriever, validate_fn, tickets, vectorizer):
    results = []
    latency_total = 0.0

    for ticket in tickets:
        issue = ticket.get("Issue", "")
        subject = ticket.get("Subject", "")
        company = ticket.get("Company", "None")

        ticket_text, company_norm = preprocess_ticket(issue, subject, company)
        query_tokens = tokenize_and_filter(ticket_text)
        request_type, product_area, _ = classify_ticket(ticket_text, company_norm)
        risk_level, _ = analyze_risk(ticket_text, request_type)
        base_action, base_reason, base_failure = decide_action(risk_level, request_type, company_norm, ticket_text)

        chunk_score = 0.0
        chunk_overlap = 0
        action = base_action
        response = ""
        sentence_scores = []
        failure_type = base_failure

        t0 = time.time()
        if action == "proceed_to_retrieval":
            expanded = [ticket_text, " ".join(query_tokens)]
            best_chunks = []
            for q in expanded:
                if not q.strip():
                    continue
                chunks = retriever.retrieve(q, product_area, top_k=3)
                best_chunks.extend(chunks)

            best_chunks.sort(key=lambda x: x["score"], reverse=True)
            seen = set()
            deduped = []
            for c in best_chunks:
                if c["content"] not in seen:
                    deduped.append(c)
                    seen.add(c["content"])

            extraction_success = False
            for chunk in deduped:
                is_valid, _, c_score, c_overlap = validate_fn(ticket_text, chunk)
                if not is_valid:
                    continue

                resp, _, scores = construct_response_from_chunk(
                    ticket_text, chunk["content"], request_type, vectorizer
                )
                if resp:
                    response = resp
                    action = "replied"
                    chunk_score = c_score
                    chunk_overlap = c_overlap
                    sentence_scores = scores
                    extraction_success = True
                    failure_type = "NONE"
                    break

            if not extraction_success:
                action = "escalated"
                failure_type = "LOW_RETRIEVAL_CONFIDENCE"

        latency_total += time.time() - t0

        if action == "escalated":
            response = "Escalated to human agent."

        confidence_score = 0.0
        if action == "replied" and request_type != "invalid":
            top_score = sentence_scores[0]["score"] if sentence_scores else 0.0
            raw = chunk_score + (chunk_overlap * 0.1) + top_score
            confidence_score = round(min(raw / 15.0, 1.0), 2)

        results.append({
            "ticket": ticket_text[:80],
            "status": action,
            "product_area": product_area,
            "request_type": request_type,
            "risk": risk_level,
            "tfidf_score": round(chunk_score, 4),
            "keyword_overlap": chunk_overlap,
            "confidence": confidence_score,
            "failure_type": failure_type,
        })

    return results, latency_total


def summarise(results, label, latency):
    total = len(results)
    replied = sum(1 for r in results if r["status"] == "replied")
    escalated = total - replied
    confidences = [r["confidence"] for r in results]
    avg_conf = sum(confidences) / total if total else 0

    return {
        "engine": label,
        "total_tickets": total,
        "replied": replied,
        "escalated": escalated,
        "reply_rate": round(replied / total, 3),
        "escalation_rate": round(escalated / total, 3),
        "total_retrieval_latency_s": round(latency, 3),
        "avg_latency_per_ticket_ms": round((latency / total) * 1000, 1) if total else 0,
        "confidence": {
            "mean": round(avg_conf, 3),
            "max": round(max(confidences), 3) if confidences else 0,
            "min": round(min(confidences), 3) if confidences else 0,
        }
    }


def main():
    print("=" * 60)
    print("Phase C Benchmarking: TF-IDF vs Semantic Retrieval")
    print("=" * 60)

    if not SEMANTIC_AVAILABLE:
        print("ERROR: sentence-transformers not installed.")
        return

    tickets = []
    with open(TICKETS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickets.append(row)

    print(f"Loaded {len(tickets)} tickets.\n")

    # --- TF-IDF ---
    print("[1/2] Building TF-IDF index...")
    tfidf = TFIDFRetriever(DATA_DIR)

    def tfidf_validate(query, chunk):
        return validate_chunk(query, chunk)

    tfidf_results, tfidf_latency = run_retrieval_pipeline(
        tfidf, tfidf_validate, tickets, tfidf.vectorizer
    )
    tfidf_summary = summarise(tfidf_results, "TF-IDF", tfidf_latency)
    print(f"  TF-IDF done: latency={tfidf_latency:.2f}s | reply_rate={tfidf_summary['reply_rate']}")

    # --- Semantic ---
    print("[2/2] Building Semantic index...")
    semantic = SemanticRetriever(DATA_DIR)

    def semantic_validate(query, chunk):
        return validate_chunk_semantic(query, chunk, tokenize_and_filter)

    semantic_results, sem_latency = run_retrieval_pipeline(
        semantic, semantic_validate, tickets, tfidf.vectorizer  # reuse TF-IDF vectorizer for sentence scoring
    )
    sem_summary = summarise(semantic_results, "Semantic (MiniLM-L6)", sem_latency)
    print(f"  Semantic done: latency={sem_latency:.2f}s | reply_rate={sem_summary['reply_rate']}")

    # --- Save summaries ---
    with open(BENCHMARK_DIR / "benchmark_tfidf.json", "w") as f:
        json.dump(tfidf_summary, f, indent=2)
    with open(BENCHMARK_DIR / "benchmark_semantic.json", "w") as f:
        json.dump(sem_summary, f, indent=2)

    # --- Side-by-side comparison CSV ---
    with open(BENCHMARK_DIR / "benchmark_comparison.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["ticket", "tfidf_status", "semantic_status", "agree",
                      "tfidf_conf", "semantic_conf", "tfidf_score", "semantic_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t, s in zip(tfidf_results, semantic_results):
            writer.writerow({
                "ticket": t["ticket"],
                "tfidf_status": t["status"],
                "semantic_status": s["status"],
                "agree": t["status"] == s["status"],
                "tfidf_conf": t["confidence"],
                "semantic_conf": s["confidence"],
                "tfidf_score": t["tfidf_score"],
                "semantic_score": s["tfidf_score"],
            })

    # --- Print comparison ---
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Metric':<35} {'TF-IDF':>12} {'Semantic':>12}")
    print("-" * 60)
    print(f"{'Reply Rate':<35} {tfidf_summary['reply_rate']:>12.3f} {sem_summary['reply_rate']:>12.3f}")
    print(f"{'Escalation Rate':<35} {tfidf_summary['escalation_rate']:>12.3f} {sem_summary['escalation_rate']:>12.3f}")
    print(f"{'Avg Latency/Ticket (ms)':<35} {tfidf_summary['avg_latency_per_ticket_ms']:>12.1f} {sem_summary['avg_latency_per_ticket_ms']:>12.1f}")
    print(f"{'Mean Confidence':<35} {tfidf_summary['confidence']['mean']:>12.3f} {sem_summary['confidence']['mean']:>12.3f}")
    print(f"{'Max Confidence':<35} {tfidf_summary['confidence']['max']:>12.3f} {sem_summary['confidence']['max']:>12.3f}")
    print("=" * 60)

    # --- Decision ---
    agree_count = sum(1 for t, s in zip(tfidf_results, semantic_results) if t["status"] == s["status"])
    agree_pct = agree_count / len(tfidf_results) * 100
    print(f"\nDecision Agreement: {agree_count}/{len(tfidf_results)} ({agree_pct:.1f}%)")

    better = "TF-IDF" if tfidf_summary["reply_rate"] >= sem_summary["reply_rate"] else "Semantic"
    print(f"\nRecommendation: {better} achieves higher reply rate on this dataset.")
    print("Per project rules: TF-IDF remains default unless Semantic strictly improves reply/confidence.")
    print("\nReports saved to benchmark/")


if __name__ == "__main__":
    main()
