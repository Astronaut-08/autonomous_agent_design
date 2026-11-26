
import json
import time
import re

from typing import TypedDict, Annotated, Literal, cast
from pathlib import Path
from pydantic import Field, BaseModel
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from tools import TOOLS
from safety import MAX_STEPS, LoopDetector, STOP_NOTES, TIMEOUT_SECONDS, is_timed_out
from logger import TrajectoryLogger
from menu_data import find_dish


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
    formatter_fallback: bool | None

class AnswerMeta(BaseModel):
    confidence: float = Field(description="Впевненість у відповіді від 0 до 1.")
    sources: list[str] = Field(description="id страв (DISH-XXX) з результатів tools, на які спирається відповідь.")

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
meta_llm = ChatOllama(model="qwen3", temperature=0, reasoning=False,
                      num_predict=200).with_structured_output(AnswerMeta)
text_llm = ChatOllama(model="qwen3", temperature=0, reasoning=False, num_predict=300)
 
# ── Вузол agent ──────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Ти асистент кафе. Допомагаєш знайти страви, перевірити наявність і порахувати замовлення. "
    "ЗАБОРОНЕНО називати страви, ціни, склад чи наявність зі своїх знань: "
    "спершу викличи відповідний інструмент і спирайся лише на його результат. "
    "Порядок: search_items -> check_availability -> calculate_order. "
    "Для розрахунку суми обов'язково використай calculate_order. "
    "Відповідай українською. "
    "Тип страви (десерти, напої, салати...) задавай параметром category, а не в query. "
    "Якщо страви з запиту немає в меню або вона недоступна (поле unavailable_matches), "
    "обов'язково повідом про це клієнту і не ігноруй позицію мовчки. "
    "У фінальній відповіді вказуй назви, ціни, наявність і час приготування, "
    "якщо вони є в результатах інструментів."
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
    "Оціни відповідь асистента за результатами інструментів. Поверни JSON з ключами "
    "'confidence' (число від 0 до 1: наскільки відповідь підтверджена результатами tools) і "
    "'sources' (список id страв DISH-XXX з результатів tools, на які спирається відповідь). "
    "Лише JSON, без іншого тексту."
)

DISH_ID_RE = re.compile(r"DISH-\d{3}")

def _unavailable_from_tools(messages) -> dict[str, str]:
    """id -> назва страв, які в результатах tools позначені як недоступні."""
    found: dict[str, str] = {}
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        try:
            data = json.loads(m.content) # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        ids = [i["id"] for i in data.get("unavailable_matches", [])]
        ids += [i["id"] for i in data.get("items", [])
                if isinstance(i, dict) and i.get("status") == "unavailable"]
        ids += data.get("unavailable", [])
        for dish_id in ids:
            dish = find_dish(dish_id)
            found[dish_id] = dish["name"] if dish else dish_id
    return found

def formatter_node(state: AgentState, config: RunnableConfig) -> dict:
    ctx: RunContext = (config.get("configurable") or {})["ctx"]
    messages = state["messages"]
    reason = state.get("stop_reason")
    last = messages[-1]
    draft = last.content if (isinstance(last, AIMessage) and last.content and not reason) else None

    tool_ids = {i for m in messages if isinstance(m, ToolMessage)
                for i in DISH_ID_RE.findall(str(m.content))}

    # 1. confidence і sources: інструкція йде ОСТАННІМ повідомленням від користувача,
    #    інакше модель може «продовжити» відповідь асистента замість виконання інструкції
    fallback_used = False
    try:
        meta = cast(AnswerMeta, meta_llm.invoke([*messages, HumanMessage(FORMATTER_PROMPT)]))
        confidence = min(max(float(meta.confidence), 0.0), 1.0)
        sources = sorted(set(meta.sources) & tool_ids) or sorted(tool_ids)
    except Exception as e:
        fallback_used = True
        print(f"[formatter] fallback: {type(e).__name__}: {str(e)[:200]}")
        confidence, sources = 0.3, sorted(tool_ids)

    # 2. answer: з повідомлення агента; після зупинки короткий підсумок
    if draft:
        answer = draft
    else:
        try:
            answer = text_llm.invoke([
                *messages,
                HumanMessage("Коротко підсумуй для клієнта українською, що вдалося з'ясувати "
                             "з результатів інструментів. Не вигадуй фактів."),
            ]).content
        except Exception:
            answer = ""
        answer = answer or "Відповідь не сформовано."

    # 3. гарантія: недоступні страви не губляться мовчки
    unavailable = _unavailable_from_tools(messages)
    if unavailable and "недоступн" not in answer.lower(): # type: ignore[arg-type]
        names = ", ".join(f"«{n}»" for n in unavailable.values())
        answer += f"\n\nУвага: зараз недоступно: {names}."

    if reason:
        answer = f"⚠️ {STOP_NOTES[reason]} {answer}"
        confidence = min(confidence, 0.3)

    payload = AgentResponse(answer=answer, confidence=confidence, sources=sources).model_dump() # type: ignore[arg-type]

    ctx.logger.log_step(state.get("step_count", 0), "formatter", str(last.content),
                        ("FALLBACK: " if fallback_used else "") + answer, []) # type: ignore[arg-type]
    return {"final_response": payload, "formatter_fallback": fallback_used}
 
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
        'formatter_fallback': None
    }
    config: RunnableConfig = {"configurable": {"ctx": ctx}, "recursion_limit": 2 * max_steps + 5}
    result = app.invoke(state_input, config=config)
    ctx.logger.save(log_path, meta={"query": query, "stop_reason": result.get("stop_reason")})
    return result