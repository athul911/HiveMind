"""Build the LangGraph ``StateGraph``.

Topology: ``START → supervisor → execute → END``.
  * ``supervisor`` chooses a route over the registered agents.
  * ``execute`` runs the routed agents (single / sequential / parallel) via the shared
    agent runtime, emitting fine-grained events through LangGraph's custom stream writer.
Checkpointing persists state after each node. Safety rails: ``recursion_limit`` and a
per-workflow token budget enforced inside the agent runtime.
"""

from __future__ import annotations

import asyncio

from hivemind.core.context import get_context, update_context
from hivemind.core.errors import BudgetExceededError
from hivemind.core.graph import events
from hivemind.core.graph.agent_runtime import run_agent_turn
from hivemind.core.graph.deps import GraphDeps
from hivemind.core.graph.state import GraphState, dict_to_msg
from hivemind.core.graph.supervisor import decide_route
from hivemind.core.llm.base import Message


def build_graph(deps: GraphDeps, checkpointer):
    from langgraph.config import get_stream_writer
    from langgraph.graph import END, START, StateGraph

    async def supervisor_node(state: GraphState) -> GraphState:
        writer = get_stream_writer()
        writer(events.node_start("supervisor").to_dict())
        plan = await decide_route(deps, state["user_message"])
        writer(events.routing_decision(plan).to_dict())
        writer(events.node_end("supervisor").to_dict())
        return {"route": plan, "iterations": state.get("iterations", 0) + 1}

    async def execute_node(state: GraphState) -> GraphState:
        writer = get_stream_writer()
        plan = state.get("route", {})
        mode = plan.get("mode", "single")
        agent_ids: list[str] = plan.get("agents", [])
        history = [dict_to_msg(d) for d in state.get("messages", [])]

        def emit(ev: events.GraphEvent) -> None:
            writer(ev.to_dict())

        # Enforce a *cumulative, per-conversation* token budget. ``tokens_used`` is
        # checkpointed, so the ceiling holds across multiple turns of the same conversation,
        # not just within one agent turn.
        budget = deps.settings.supervisor_token_budget
        used_before = state.get("tokens_used", 0)
        if used_before >= budget:
            raise BudgetExceededError(
                "Per-conversation token budget exhausted.", tokens_used=used_before
            )
        remaining = budget - used_before

        if mode == "parallel" and len(agent_ids) > 1:
            final, outputs, tokens = await _run_parallel(deps, agent_ids, history, emit, remaining)
        elif mode == "conditional" and len(agent_ids) >= 1:
            final, outputs, tokens = await _run_conditional(
                deps, plan, agent_ids, history, emit, remaining
            )
        else:  # single or sequential
            final, outputs, tokens = await _run_sequential(
                deps, agent_ids, history, emit, remaining
            )

        emit(events.message("assistant", final))
        emit(events.done(final))
        return {
            "final_response": final,
            "agent_outputs": outputs,
            "tokens_used": used_before + tokens,
        }

    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("execute", execute_node)
    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "execute")
    graph.add_edge("execute", END)
    return graph.compile(checkpointer=checkpointer)


async def _run_one_agent(
    deps: GraphDeps, agent_id: str, history: list[Message], emit, budget: int
) -> tuple[str, dict, int]:
    agent = deps.agents.get(agent_id)
    prepared = deps.agent_factory.prepare(agent)
    token = update_context(agent_id=agent_id)
    emit(events.node_start("agent", agent_id=agent_id))
    try:
        ctx = get_context()
        final, _appended, usage = await run_agent_turn(
            deps=deps,
            system_prompt=prepared.effective_system_prompt,
            tool_names=list(agent.tool_names),
            llm_config=prepared.llm_config,
            conversation=history,
            ctx=ctx,
            emit=emit,
            token_budget=budget,
        )
    finally:
        emit(events.node_end("agent", agent_id=agent_id))
        from hivemind.core.context import reset_context

        reset_context(token)
    output = {"agent_id": agent_id, "text": final}
    return final, output, usage.input_tokens + usage.output_tokens


async def _run_sequential(
    deps: GraphDeps, agent_ids: list[str], history: list[Message], emit, remaining: int
) -> tuple[str, list[dict], int]:
    outputs: list[dict] = []
    convo = list(history)
    final = ""
    total = 0
    for agent_id in agent_ids:
        if remaining - total <= 0:
            raise BudgetExceededError("Token budget exhausted mid-pipeline.", tokens_used=total)
        final, output, tokens = await _run_one_agent(
            deps, agent_id, convo, emit, remaining - total
        )
        outputs.append(output)
        total += tokens
        # Feed this agent's answer forward to the next agent in the pipeline.
        convo = [*convo, Message(role="assistant", content=final)]
    return final, outputs, total


async def _run_parallel(
    deps: GraphDeps, agent_ids: list[str], history: list[Message], emit, remaining: int
) -> tuple[str, list[dict], int]:
    # Each branch is individually capped at the remaining budget (they run concurrently).
    results = await asyncio.gather(
        *(_run_one_agent(deps, aid, history, emit, remaining) for aid in agent_ids)
    )
    outputs = [r[1] for r in results]
    total = sum(r[2] for r in results)
    aggregated = "\n\n".join(f"[{o['agent_id']}]\n{o['text']}" for o in outputs)
    return aggregated, outputs, total


async def _run_conditional(
    deps: GraphDeps, plan: dict, agent_ids: list[str], history: list[Message], emit, remaining: int
) -> tuple[str, list[dict], int]:
    """Run the primary agent, then branch on its result.

    The supervisor's plan carries a ``condition`` (natural language). After the primary
    agent answers, a cheap LLM check decides whether the condition holds; if so, the
    remaining agent(s) run sequentially with the primary's output in context. This realizes
    "conditional branching based on intermediate results."
    """
    from hivemind.core.graph.conditional import evaluate_condition

    primary, *fallback = agent_ids
    final, output, tokens = await _run_one_agent(deps, primary, history, emit, remaining)
    outputs = [output]
    condition = plan.get("condition", "")

    if fallback and condition:
        emit(events.GraphEvent("condition_check", {"condition": condition}))
        triggered, eval_tokens = await evaluate_condition(deps, condition, final)
        tokens += eval_tokens
        emit(events.GraphEvent("condition_result", {"condition": condition, "met": triggered}))
        if triggered and remaining - tokens > 0:
            convo = [*history, Message(role="assistant", content=final)]
            f2, more_outputs, t2 = await _run_sequential(
                deps, fallback, convo, emit, remaining - tokens
            )
            outputs.extend(more_outputs)
            tokens += t2
            final = f2
    return final, outputs, tokens
