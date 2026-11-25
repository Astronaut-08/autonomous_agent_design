
import json
import time

from typing import TypedDict, Annotated, Literal
from pathlib import Path
from pydantic import Field, BaseModel
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from tools import TOOLS
from safety import MAX_STEPS, LoopDetector, STOP_NOTES, TIMEOUT_SECONDS, is_timed_out
from logger import TrajectoryLogger


# ── State ────────────────────────────────────────────────────────
class AgentResponse(BaseModel):
    answer: str = Field(description="Фінальна відповідь клієнту українською.")
    confidence: float = Field(ge=0, le=1, description="Впевненість у відповіді від 0 до 1.")
    sources: list[str] = Field(
        description="id страв (DISH-001) або назви tools, на яких базується відповідь."
    )

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    step_count: int
    final_response: dict | None
    stop_reason: str | None 

@dataclass
class RunContext:
    '''Об'єкти, що живуть один запуск агента.'''
    logger: TrajectoryLogger
    detector: LoopDetector
    start: float
    max_steps: int
    timeout: float
 
# ── LLM з tools ──────────────────────────────────────────────────
llm = ChatOllama(model='qwen3', temperature=0, reasoning=False)
agent_llm = llm.bind_tools(TOOLS)
formated_llm = llm.with_structured_output(AgentResponse)

for t in TOOLS:
    t.handle_validation_error = True
 
# ── Вузол agent ──────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Ти асистент кафе. Допомагаєш знайти страви, перевірити наявність і порахувати замовлення. "
    "ЗАБОРОНЕНО називати страви, ціни, склад чи наявність зі своїх знань: "
    "спершу викличи відповідний інструмент і спирайся лише на його результат. "
    "Порядок: search_items -> check_availability -> calculate_order. "
    "Для розрахунку суми обов'язково використай calculate_order. "
    "Відповідай українською."
    "Тип страви (десерти, напої, салати...) задавай параметром category, а не в query. "
    "Якщо страви з запиту немає в меню або вона недоступна (поле unavailable_matches), "
    "обов'язково повідом про це клієнту і не ігноруй позицію мовчки. "
)

def _stop(ctx: RunContext, step: int, reason: str, dropped=None) -> dict:
    '''Зупинка: замість виклику, який не виконаємо, додаємо службове повідомлення.'''
    ctx.logger.log_step(step, "agent", "", f"STOP: {reason}", dropped or [])
    return {"messages": [AIMessage(content=f"[СТОП] {STOP_NOTES[reason]}")],
            "step_count": step, "stop_reason": reason}

def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """LLM приймає рішення: викликати tool чи відповісти."""
    ctx: RunContext = (config.get("configurable") or {})["ctx"]
    step = state.get("step_count", 0)

     # 1. timeout перевіряємо ДО дорогого виклику LLM
    if is_timed_out(ctx.start, ctx.timeout):
        return _stop(ctx, step, "timeout")
    
    response = agent_llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
    step += 1
    calls = [{"name": c["name"], "args": c["args"]} for c in (response.tool_calls or [])]

    if calls:
        looped = [ctx.detector.check(c["name"], c["args"]) for c in calls]
        # 2. зациклення, 3. ліміт кроків
        if any(looped):
            return _stop(ctx, step, "loop_detected", calls)
        if step >= ctx.max_steps:
            return _stop(ctx, step, "max_steps", calls)

    ctx.logger.log_step(step, "agent", state["messages"][-1].content,
                        str(response.content) or "", calls)
    
    return {"messages": [response], "step_count": step}

# ── Вузол tools ──────────────────────────────────────────────────
tool_executor = ToolNode(TOOLS)

def tools_node(state: AgentState, config: RunnableConfig) -> dict:
    ctx: RunContext = (config.get("configurable") or {})["ctx"]
    calls = [{"name": c["name"], "args": c["args"]} for c in state["messages"][-1].tool_calls]
    result = tool_executor.invoke(state, config)
    output = " | ".join(str(m.content) for m in result["messages"])
    ctx.logger.log_step(state["step_count"], "tools", json.dumps(calls, ensure_ascii=False),
                        output, calls)
    return result


# ── Вузол formatter ──────────────────────────────────────────────────
FORMATTER_PROMPT = (
    "На основі діалогу сформуй фінальну відповідь у форматі JSON з ключами: "
    "'answer' (рядок українською), 'confidence' (число від 0 до 1), 'sources' (список рядків з id страв або назв tools). "
    "Використовуй лише факти, отримані з результатів tools. Не додавай жодного тексту поза JSON."
    "У 'answer' вказуй назви страв і їхні ціни. Якщо щось не знайдено або недоступне, скажи про це. "
    "У 'sources' перелічи id страв з результатів tools; якщо нічого не знайдено, confidence не більше 0.5."
)

def formatter_node(state: AgentState) -> dict:
    try:
        result = formated_llm.invoke([SystemMessage(FORMATTER_PROMPT), *state["messages"]])
        payload = result.model_dump() if isinstance(result, BaseModel) else dict(result)
    except Exception as e:
        print(f"[formatter] fallback: {type(e).__name__}: {e}")   # щоб бачити справжню причину
        fallback = llm.invoke([                                    # llm, а не agent_llm
            SystemMessage("Сформуй коротку відповідь українською на основі діалогу."),
            *state["messages"],
        ])
        payload = {
            "answer": fallback.content or "Не вдалося сформувати відповідь.",
            "confidence": 0.3,
            "sources": [],
        }

    reason = state.get("stop_reason")
    if reason:
        payload["answer"] = f"⚠️ {STOP_NOTES[reason]} {payload['answer']}"
        payload["confidence"] = min(payload["confidence"], 0.3)
    return {"final_response": payload}
 
# ── Router ───────────────────────────────────────────────────────
def should_continue(state: AgentState) -> Literal["tools", "formatter"]:
    if state.get("stop_reason"):
        return "formatter"
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "formatter"
 
# ── Граф ─────────────────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)
graph.add_node("formatter", formatter_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "formatter": "formatter"})
graph.add_edge("tools", "agent")
graph.add_edge("formatter", END)

app = graph.compile()
 
# ── Запуск ───────────────────────────────────────────────────────
def run_agent(query: str, *, max_steps: int = MAX_STEPS, timeout: float = TIMEOUT_SECONDS,
                log_path: str = f'{Path(__file__).resolve().parent}/trajectory.json') -> dict:
    ctx = RunContext(TrajectoryLogger(), LoopDetector(), time.monotonic(), max_steps, timeout)
    state_input: AgentState = {
        "messages": [HumanMessage(content=query)],
        "step_count": 0, "final_response": None, "stop_reason": None,
    }
    config: RunnableConfig = {"configurable": {"ctx": ctx}, "recursion_limit": 2 * max_steps + 5}
    result = app.invoke(state_input, config=config)
    ctx.logger.save(log_path, meta={"query": query, "stop_reason": result.get("stop_reason")})
    return result