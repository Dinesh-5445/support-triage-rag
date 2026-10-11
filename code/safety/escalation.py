"""
Escalation Manager — V2
Deterministic decision engine. LLM generation cannot override these decisions.

Supported outcomes:
  ANSWER (status="replied")
  PARTIAL_ANSWER_ESCALATE (status="partial_answer_escalate")
  ESCALATE (status="escalated")
  OUT_OF_SCOPE (status="out_of_scope")
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision import decide_action as legacy_decide_action


class EscalationManager:
    """
    Pre-retrieval and post-generation decision engine.
    Deterministic. Cannot be overridden by LLM output.
    """

    def evaluate_pre_retrieval(self, risk_level: str, request_type: str,
                                company: str, text: str) -> dict:
        """
        Runs before retrieval. Determines whether to:
         - proceed to retrieval
         - immediately escalate (high risk, OOD, etc.)
         - mark as out_of_scope
        """
        # OUT_OF_SCOPE: invalid domain / company is None with non-domain request
        if request_type == "invalid":
            return {
                "action": "out_of_scope",
                "reason": "Request does not match any supported domain (HackerRank, Claude, Visa).",
                "failure_type": "OUT_OF_SCOPE",
            }

        action, reason, failure_type = legacy_decide_action(
            risk_level, request_type, company, text
        )
        return {
            "action": action,  # "escalated", "replied", "proceed_to_retrieval"
            "reason": reason,
            "failure_type": failure_type,
        }

    def evaluate_post_generation(self, grounding_result: dict,
                                  max_retries_reached: bool = False) -> dict:
        """
        Runs after generation + grounding.
        Uses grounding_status field from GroundingValidator.

        Returns one of:
          replied               — fully grounded answer
          partial_answer_escalate — partial grounding (PARTIAL)
          retry                 — not grounded, retry available
          escalated             — not grounded, max retries reached
        """
        grounding_status = grounding_result.get("grounding_status", "UNGROUNDED")
        is_grounded = grounding_result.get("is_grounded", False)
        reason = grounding_result.get("reason", "")

        # Fully grounded → ANSWER
        if is_grounded and grounding_status == "GROUNDED":
            return {
                "action": "replied",
                "reason": "Answer is grounded. " + reason,
                "failure_type": "NONE",
            }

        # Partially grounded → PARTIAL_ANSWER_ESCALATE
        if is_grounded and grounding_status == "PARTIAL":
            return {
                "action": "partial_answer_escalate",
                "reason": (
                    "Evidence supports a partial answer but is insufficient for complete resolution. "
                    + reason
                ),
                "failure_type": "PARTIAL_GROUNDING",
            }

        # Not grounded
        if max_retries_reached:
            return {
                "action": "escalated",
                "reason": "Max retrieval attempts reached. Grounding failed: " + reason,
                "failure_type": "GROUNDING_FAILURE",
            }

        # Retry available
        return {
            "action": "retry",
            "reason": "Ungrounded answer — triggering retrieval retry. " + reason,
            "failure_type": "GROUNDING_FAILURE",
        }
