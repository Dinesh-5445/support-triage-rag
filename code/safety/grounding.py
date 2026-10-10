"""
Grounding Validator — V2
Validates that the evidence retrieved actually supports the generated/extracted response.
Tracks: evidence_available, evidence_relevance, token_overlap, grounding_status.
"""


class GroundingValidator:
    """
    Performs evidence-based grounding validation.

    Grounding is assessed by:
    1. Whether any evidence was retrieved (evidence_available)
    2. Whether the top chunk has sufficient relevance score (evidence_relevance)
    3. Token overlap between the response and retrieved evidence (overlap check)
    4. Whether the response itself signals insufficient evidence (self-reported failure)

    This is NOT a simple confidence-threshold check. It uses multiple orthogonal signals.
    """

    # Minimum cosine/semantic score for a chunk to be considered relevant
    RELEVANCE_THRESHOLD = 0.20
    # Minimum token overlap between response and evidence
    MIN_TOKEN_OVERLAP = 2

    def __init__(self, grounding_threshold=0.20):
        self.grounding_threshold = grounding_threshold

    def validate(self, generated_answer: str, retrieved_chunks: list, confidence_score: float) -> dict:
        """
        Returns a grounding result dict with:
          is_grounded: bool
          reason: str
          evidence_available: bool
          evidence_relevance: float  (top chunk score)
          token_overlap: int
          unsupported_claims: bool  (detected via self-report)
          grounding_status: str     (GROUNDED | PARTIAL | UNGROUNDED)
        """
        result = {
            "is_grounded": False,
            "reason": "",
            "evidence_available": False,
            "evidence_relevance": 0.0,
            "token_overlap": 0,
            "unsupported_claims": False,
            "grounding_status": "UNGROUNDED",
        }

        # --- Check 1: Evidence available ---
        if not retrieved_chunks:
            result["reason"] = "No evidence retrieved."
            result["grounding_status"] = "UNGROUNDED"
            return result

        result["evidence_available"] = True
        top_chunk = retrieved_chunks[0]
        top_score = top_chunk.get("score", 0.0)
        result["evidence_relevance"] = round(float(top_score), 4)

        # --- Check 2: Evidence relevance ---
        if top_score < self.RELEVANCE_THRESHOLD:
            result["reason"] = (
                f"Top evidence score {top_score:.3f} is below relevance threshold "
                f"{self.RELEVANCE_THRESHOLD}."
            )
            result["grounding_status"] = "UNGROUNDED"
            return result

        # --- Check 3: Token overlap between response and evidence ---
        evidence_text = " ".join(c.get("content", "") for c in retrieved_chunks)
        response_tokens = set(generated_answer.lower().split())
        evidence_tokens = set(evidence_text.lower().split())
        overlap = len(response_tokens & evidence_tokens)
        result["token_overlap"] = overlap

        # --- Check 4: Self-reported failure phrases ---
        failure_phrases = [
            "insufficient evidence",
            "cannot construct",
            "escalat",
            "no relevant information",
            "unable to find",
        ]
        self_reported_fail = any(p in generated_answer.lower() for p in failure_phrases)
        result["unsupported_claims"] = self_reported_fail

        if self_reported_fail:
            result["reason"] = "Response self-reported insufficient evidence."
            result["grounding_status"] = "UNGROUNDED"
            return result

        # --- Partial grounding: evidence is relevant but overlap is very low ---
        if overlap < self.MIN_TOKEN_OVERLAP:
            result["reason"] = (
                f"Evidence is relevant (score={top_score:.3f}) but token overlap with "
                f"response is {overlap} (min={self.MIN_TOKEN_OVERLAP}). Partial grounding."
            )
            result["grounding_status"] = "PARTIAL"
            result["is_grounded"] = True  # Allow partial answer to proceed
            return result

        # --- Fully grounded ---
        result["is_grounded"] = True
        result["grounding_status"] = "GROUNDED"
        result["reason"] = (
            f"Evidence score={top_score:.3f}, token_overlap={overlap}. Grounding passed."
        )
        return result
