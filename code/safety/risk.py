import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification import analyze_risk as legacy_analyze_risk

class RiskAnalyzer:
    def __init__(self):
        pass

    def analyze(self, text, request_type):
        risk_level, mode = legacy_analyze_risk(text, request_type)
        return {
            "risk_level": risk_level,
            "mode": mode
        }
