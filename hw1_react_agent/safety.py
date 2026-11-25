import hashlib, json, time


MAX_STEPS = 15
TIMEOUT_SECONDS = 120
LOOP_MAX_REPEATS = 3

STOP_NOTES = {
    "max_steps": "Досягнуто ліміту кроків, відповідь часткова.",
    "timeout": "Перевищено ліміт часу, відповідь часткова.",
    "loop_detected": "Виявлено зациклення (повторюваний виклик інструменту), відповідь часткова.",
}


class LoopDetector:
    """Детекція зациклення: той самий tool з тими самими аргументами N разів поспіль."""
 
    def __init__(self, max_repeats: int = LOOP_MAX_REPEATS):
        self.recent_calls: list[str] = []
        self.max_repeats = max_repeats
 
    def check(self, tool_name: str, args: dict) -> bool:
        """Перевірити, чи виклик повторюється.
 
        Returns:
            True якщо зациклення виявлено.
        """
        payload = json.dumps({'tool': tool_name, 'args': args}, 
                            sort_keys=True, ensure_ascii=False, default=str)
        self.recent_calls.append(hashlib.md5(payload.encode()).hexdigest())
        last_n = self.recent_calls[-self.max_repeats:]
        return len(last_n) == self.max_repeats and len(set(last_n)) == 1

def is_timed_out(start: float, timeout: float = TIMEOUT_SECONDS) -> bool:
    return time.monotonic() - start >= timeout
