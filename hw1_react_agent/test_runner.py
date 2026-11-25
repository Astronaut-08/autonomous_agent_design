import json
import tempfile
import time
from collections import Counter
from pathlib import Path

from agent import run_agent

BASE_DIR = Path(__file__).parent

# expected_tools: tools, які агент мав викликати
# expected_keywords: список груп; у відповіді має бути хоча б одне слово з кожної групи
TEST_CASES = [
    {
        "id": "TC-001",
        "complexity": "simple",
        "query": "Які десерти є в меню?",
        "expected": "Два десерти: тарт татен DISH-009 (155 грн) і чіа-пудинг DISH-010 (145 грн).",
        "expected_tools": ["search_items"],
        "expected_keywords": [["DISH-009", "тарт"], ["DISH-010", "чіа"]],
    },
    {
        "id": "TC-002",
        "complexity": "simple",
        "query": "Які безлактозні напої коштують до 100 грн?",
        "expected": "Лимонад DISH-011 (95 грн) і лате на безлактозному молоці DISH-012 (85 грн).",
        "expected_tools": ["search_items"],
        "expected_keywords": [["DISH-011", "лимонад"], ["DISH-012", "лате"]],
    },
    {
        "id": "TC-003",
        "complexity": "medium",
        "query": "Порадь веганський салат до 250 грн і скажи, чи він доступний",
        "expected": "Теплий салат з кіноа DISH-004 (240 грн), він доступний.",
        "expected_tools": ["search_items", "check_availability"],
        "expected_keywords": [["DISH-004", "кіноа"], ["240"], ["доступн"]],
    },
    {
        "id": "TC-004",
        "complexity": "complex",
        "query": "Хочу 2 чіа-пудинги і лимонад. Скільки це коштує і скільки чекати?",
        "expected": "Разом 385 грн (2×145 + 95), орієнтовний час приготування 5 хв.",
        "expected_tools": ["search_items", "check_availability", "calculate_order"],
        "expected_keywords": [["385"], ["5 хв"]],
    },
    {
        "id": "TC-005",
        "complexity": "complex",
        "query": "Хочу замовити філе лосося і борщ. Порахуй суму.",
        "expected": "Лосось (DISH-006) зараз недоступний; агент повідомляє про це і не рахує "
                    "замовлення з ним (або пропонує заміну).",
        "expected_tools": ["search_items"],
        "expected_keywords": [["лосос"], ["недоступн", "немає", "відсутн"]],
    },
    # Додатково, для аналізу впливу max_steps
    {
        "id": "TC-006",
        "complexity": "complex",
        "query": "Хочу 2 чіа-пудинги і лимонад. Скільки це коштує і скільки чекати?",
        "expected": "Часткова відповідь із повідомленням про ліміт кроків.",
        "expected_tools": ["search_items"],
        "expected_keywords": [["ліміт"]],
        "run_kwargs": {"max_steps": 2},
    },
]

STATUS_LABEL = {
    "success": "✅ success",
    "partial": "⚠️ partial",
    "failed": "❌ failed",
    "error": "💥 error",
}


def collect_tool_calls(messages) -> list[str]:
    names = []
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            names.append(call.get("name", "unknown"))
    return names


def evaluate(tc: dict, answer: str, tool_names: list[str], stop_reason: str | None):
    used = set(tool_names)
    text = answer.lower()
    missing_tools = [t for t in tc["expected_tools"] if t not in used]
    missing_kw = [g for g in tc["expected_keywords"]
                  if not any(k.lower() in text for k in g)]

    total = len(tc["expected_tools"]) + len(tc["expected_keywords"])
    failed = len(missing_tools) + len(missing_kw)

    if stop_reason:
        status = "partial"
    elif failed == 0:
        status = "success"
    elif failed < total:
        status = "partial"
    else:
        status = "failed"
    return status, missing_tools, missing_kw


def run_case(tc: dict, tmp_dir: str) -> tuple[dict, dict | None]:
    log_path = Path(tmp_dir) / f"{tc['id']}.json"
    start = time.monotonic()

    try:
        result = run_agent(tc["query"], log_path=str(log_path), **tc.get("run_kwargs", {}))
        final = result["final_response"] or {}
        actual = final.get("answer", "")
        stop_reason = result.get("stop_reason")
        steps = result["step_count"]
        tool_names = collect_tool_calls(result["messages"])
        status, missing_tools, missing_kw = evaluate(tc, actual, tool_names, stop_reason)
        extra = {"confidence": final.get("confidence"), "sources": final.get("sources", [])}
    except Exception as e:
        actual, stop_reason, steps, tool_names = f"Error: {type(e).__name__}: {e}", None, -1, []
        status, missing_tools, missing_kw = "error", [], []
        extra = {"confidence": None, "sources": []}

    elapsed_ms = int((time.monotonic() - start) * 1000)

    record = {
        "test_id": tc["id"],
        "complexity": tc["complexity"],
        "query": tc["query"],
        "expected": tc["expected"],
        "actual": actual[:500],
        **extra,
        "status": status,
        "steps": steps,
        "tool_calls": tool_names,
        "elapsed_ms": elapsed_ms,
        "stop_reason": stop_reason,
        "max_steps_reached": stop_reason == "max_steps",
        "timeout_reached": stop_reason == "timeout",
        "loop_detected": stop_reason == "loop_detected",
        "missing_tools": missing_tools,
        "missing_keywords": missing_kw,
    }

    trajectory = None
    if log_path.exists():
        trajectory = json.loads(log_path.read_text(encoding="utf-8"))
    return record, trajectory


def format_tools(names: list[str]) -> str:
    if not names:
        return "—"
    return ", ".join(f"{n} × {c}" for n, c in Counter(names).items())


def main() -> None:
    results, trajectories = [], {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        for tc in TEST_CASES:
            print(f"Running {tc['id']}: {tc['query'][:60]}...")
            record, trajectory = run_case(tc, tmp_dir)
            results.append(record)
            if trajectory:
                trajectories[tc["id"]] = trajectory

    (BASE_DIR / "test_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (BASE_DIR / "trajectory.json").write_text(
        json.dumps(trajectories, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nResults saved: {len(results)} test cases\n")

    # Таблиця для розділу README
    print("| Test ID | Складність | Кроків | Tool calls | Час, мс | Статус |")
    print("|---------|------------|--------|------------|---------|--------|")
    for r in results:
        print(f"| {r['test_id']} | {r['complexity']} | {r['steps']} | "
              f"{format_tools(r['tool_calls'])} | {r['elapsed_ms']} | {STATUS_LABEL[r['status']]} |")


if __name__ == "__main__":
    main()