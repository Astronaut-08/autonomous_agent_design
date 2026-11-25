import json
import time
from datetime import datetime, timezone
 
class TrajectoryLogger:
    """Логування траєкторії виконання агента."""
 
    def __init__(self):
        self.steps = []
        self.start_time = time.monotonic()
        self._mark = self.start_time # кінець попереднього кроку
 
    def log_step(self, step_num: int, node: str,
                 input_data: str, output_data: str,
                 tool_calls: list | None = None):
        now = time.monotonic()
        self.steps.append({
            'step': step_num,
            'node': node,
            'input': str(input_data)[:500],
            'output': str(output_data)[:500],
            'tool_calls': tool_calls or [],
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'elapsed_ms': int((now - self._mark) * 1000), # тривалість кроку
        }) 
        self._mark = now
 
    def save(self, filepath: str, meta: dict | None = None):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                **(meta or {}),
                'total_steps': len(self.steps),
                'total_time_ms': int((time.monotonic() - self.start_time) * 1000),
                'trajectory': self.steps,
            }, f, ensure_ascii=False, indent=2)
        print(f'Trajectory saved: {filepath} ({len(self.steps)} steps)')
 
# Використання:
logger = TrajectoryLogger()
logger.log_step(1, 'agent', 'user query', 'tool call: get_weather', ['get_weather'])
logger.log_step(2, 'tools', 'get_weather(Kyiv)', 'Kyiv: 15°C')
logger.save('trajectory.json')
