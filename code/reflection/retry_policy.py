class RetryPolicy:
    def __init__(self, max_attempts=2):
        self.max_attempts = max_attempts

    def should_retry(self, current_attempt):
        return current_attempt < self.max_attempts
