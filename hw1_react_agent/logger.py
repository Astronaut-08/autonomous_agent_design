import json
import time
from datetime import datetime, timezone


class TrajectoryLogger:
    """Логування траєкторії виконання агента."""

    def __init__(self):
        self.steps: list[dict] = []
        self.start_time = time.monotonic()
        self._mark = self.start_time          # кінець попереднього кроку

    def log_step(self, step_number: int, node_name: str,
                 input_data, output_data, tool_calls: list | None = None):
        now = time.monotonic()
        self.steps.append({
            "step_number": step_number,
            "node_name": node_name,
            "input": str(input_data)[:500],
            "output": str(output_data)[:500],
            "tool_calls": tool_calls or [],
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "duration_ms": int((now - self._mark) * 1000),   # тривалість саме цього кроку
        })
        self._mark = now

    def save(self, filepath: str, meta: dict | None = None):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                **(meta or {}),
                "log_entries": len(self.steps),
                "total_time_ms": int((time.monotonic() - self.start_time) * 1000),
                "trajectory": self.steps,
            }, f, ensure_ascii=False, indent=2)
        print(f"Trajectory saved: {filepath} ({len(self.steps)} log entries)")