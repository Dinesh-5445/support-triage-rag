import csv
import json
import time
import subprocess
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_FILE = os.path.join(BASE_DIR, "support_tickets", "baseline_output.csv")
MODULAR_FILE = os.path.join(BASE_DIR, "support_tickets", "output.csv")
METRICS_FILE = os.path.join(BASE_DIR, "results", "metrics.json")

def load_csv(path):
    with open(path, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    print("Running baseline evaluation (all tickets)...")
    try:
        subprocess.run(["python", "code/run_baseline.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running run_baseline.py: {e}")
        return

    print("Running V2 Modular RAG evaluation (all tickets)...")
    start_time = time.time()
    try:
        subprocess.run(["python", "code/main.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running main.py: {e}")
        return
    retrieval_latency = time.time() - start_time

    if not os.path.exists(BASELINE_FILE) or not os.path.exists(MODULAR_FILE):
        print("Output files missing!")
        return

    baseline_data = load_csv(BASELINE_FILE)
    modular_data = load_csv(MODULAR_FILE)

    if len(baseline_data) != len(modular_data):
        print(f"WARNING: Population mismatch! Baseline: {len(baseline_data)}, Modular: {len(modular_data)}")

    total_tickets = len(modular_data)
    
    # Outcomes
    answer_count = 0
    partial_count = 0
    escalate_count = 0
    out_of_scope_count = 0
    
    # Parity tracking
    matching_decisions = 0
    divergent_replies = 0
    divergent_escalations = 0
    
    # Internal counts
    retrieval_retry_count = 0
    total_retrieval_attempts = 0
    
    retrieval_mode_set = set()
    reranker_used_count = 0

    for idx, mod_row in enumerate(modular_data):
        mod_status = mod_row["status"]
        
        # Outcome counting
        if mod_status == "replied":
            answer_count += 1
        elif mod_status == "partial_answer_escalate":
            partial_count += 1
        elif mod_status == "escalated":
            escalate_count += 1
        elif mod_status == "out_of_scope":
            out_of_scope_count += 1
            
        # V2 specific stats
        attempts = int(mod_row.get("retrieval_attempts", 1))
        total_retrieval_attempts += attempts
        if attempts > 1:
            retrieval_retry_count += 1
            
        retrieval_mode = mod_row.get("retrieval_mode", "unknown")
        retrieval_mode_set.add(retrieval_mode)
        
        if mod_row.get("reranker_used", "False") == "True":
            reranker_used_count += 1

        # Parity calculation (only on matching index length)
        if idx < len(baseline_data):
            base_row = baseline_data[idx]
            base_status = base_row["status"]
            
            # Map V2 specific statuses back to V1 equivalents for parity comparison
            # V1 only had 'replied' or 'escalated'
            mod_v1_equivalent = mod_status
            if mod_status == "partial_answer_escalate" or mod_status == "out_of_scope":
                mod_v1_equivalent = "escalated"
                
            if base_status == mod_v1_equivalent:
                matching_decisions += 1
            else:
                if mod_v1_equivalent == "replied" and base_status == "escalated":
                    divergent_replies += 1
                elif mod_v1_equivalent == "escalated" and base_status == "replied":
                    divergent_escalations += 1

    decision_parity_percentage = (matching_decisions / len(baseline_data)) * 100 if baseline_data else 0.0

    metrics = {
        "total_tickets": total_tickets,
        "answer_count": answer_count,
        "partial_answer_escalate_count": partial_count,
        "escalate_count": escalate_count,
        "out_of_scope_count": out_of_scope_count,
        "high_risk_count": len([r for r in modular_data if r.get("failure_type") == "HIGH_RISK"]),
        "retrieval_retry_count": retrieval_retry_count,
        "average_retrieval_attempts": round(total_retrieval_attempts / total_tickets, 2) if total_tickets else 1.0,
        "decision_parity_percentage": round(decision_parity_percentage, 2),
        "divergent_reply_count": divergent_replies,
        "divergent_escalation_count": divergent_escalations,
        "total_latency_seconds": round(retrieval_latency, 2),
        "retrieval_mode": list(retrieval_mode_set)[0] if len(retrieval_mode_set) == 1 else list(retrieval_mode_set),
        "reranker_active": reranker_used_count > 0,
        "retrieval_quality_metrics": "unavailable" # No gold labels to measure Recall@K etc.
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {METRICS_FILE}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
