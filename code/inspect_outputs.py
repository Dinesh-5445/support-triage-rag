import csv
from collections import Counter
import sys, os
sys.path.insert(0, os.path.abspath("code"))

def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

baseline = load_csv("support_tickets/baseline_output.csv")
output = load_csv("support_tickets/output.csv")

print(f"baseline_output.csv rows: {len(baseline)}")
print(f"output.csv rows: {len(output)}")
if baseline:
    print(f"baseline columns: {list(baseline[0].keys())}")
if output:
    print(f"output columns: {list(output[0].keys())}")

bl_status = Counter(r["status"] for r in baseline)
out_status = Counter(r["status"] for r in output)
print(f"Baseline status distribution: {dict(bl_status)}")
print(f"Output status distribution: {dict(out_status)}")
