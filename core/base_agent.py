"""
Base Agent abstract class for GeneAgent.
"""
import logging
import time
from typing import Any

from core.config import Settings
from core.llm import LLMClient


class BaseAgent:
    def __init__(self, name: str, settings: Settings | None = None):
        self.name = name
        self.logger = logging.getLogger(f"GeneAgent.{name}")
        self.settings = settings or Settings()
        self.llm: LLMClient | None = None
        self._initialized = False
        # Messaging (Phase 1) — optional, backward-compatible


        self._message_bus = None  # set via attach_bus()

    async def _initialize(self):
        if self._initialized:
            return
        if self.settings.llm.is_valid():
            self.llm = LLMClient(self.settings.llm)
        self._initialized = True

    async def _shutdown(self):
        self._initialized = False

    # Messaging API (requires attach_bus before use)



    def attach_bus(self, bus) -> None:
        """Attach this agent to a MessageBus for multi-agent communication."""
        from core.messaging import MessageBus as _MessageBus
        self._message_bus: _MessageBus | None = bus

    def _has_bus(self) -> bool:
        return self._message_bus is not None

    async def send_message(
        self, receiver: str, msg_type, payload: dict,
        gene_symbol: str | None = None,
        round_id: str = "", debate_id: str = "",
        reply_to: str | None = None, priority: int = 0,
    ) -> str:
        """Send a typed message to another agent or broadcast.

        Returns the msg_id of the sent message.
        """
        if not self._has_bus():
            return ""
        from core.messaging import AgentMessage as _AgentMessage
        msg = _AgentMessage(
            sender=self.name,
            receiver=receiver,
            msg_type=msg_type,
            payload=payload,
            gene_symbol=gene_symbol,
            round_id=round_id,
            debate_id=debate_id,
            reply_to=reply_to,
            priority=priority,
        )
        await self._message_bus.send(msg)
        return msg.msg_id

    async def receive_messages(self, timeout: float = 0.3) -> list:
        """Drain all queued messages from this agent's inbox."""
        if not self._has_bus():
            return []
        return await self._message_bus.receive_all(self.name, timeout=timeout)

    async def wait_for_message(self, msg_type, timeout: float = 60.0):
        """Block until a specific message type arrives.  Returns None on timeout."""
        if not self._has_bus():
            return None
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = await self._message_bus.receive(self.name, timeout=1.0)
            if msg and msg.msg_type == msg_type:
                return msg
        return None

    async def broadcast(self, msg_type, payload: dict,
                        round_id: str = "", **kwargs) -> str:
        """Send a message to all registered agents."""
        return await self.send_message(
            receiver="broadcast", msg_type=msg_type,
            payload=payload, round_id=round_id, **kwargs,
        )

    # Core reasoning loop (preserved)



    async def autonomous_thinking(
        self, prompt: str, state: dict[str, Any] | None = None,
        system_message: str | None = None, max_iterations: int = 10,
        tool_call_required: bool = False,
    ) -> dict[str, Any]:
        """Autonomous reasoning loop with optional tool-call enforcement.

        Args:
            tool_call_required: If True and agent has _local_tools, the first
                response MUST include at least one tool call.  If the LLM skips
                tools, the prompt is re-sent with a stronger directive.
        """
        if not self.llm:
            return {"response": "[ERROR] LLM not configured", "tool_calls": [], "tool_results": [], "iterations": 0}

        messages = []
        if system_message or hasattr(self, '_get_system_prompt'):
            sp = system_message
            if not sp and hasattr(self, '_get_system_prompt'):
                sp = self._get_system_prompt(state, system_message)
            if sp:
                messages.append({"role": "system", "content": sp})

        ctx = ""
        if state:
            ctx = f"\n\nCurrent context:\n{state}"
        messages.append({"role": "user", "content": prompt + ctx})

        # Use non-streaming invoke for LLM-only agents (more reliable)


        tools = []
        if hasattr(self, '_local_tools'):
            for t in self._local_tools:
                tools.append({"type": "function", "function": t.get("function", t)})

        all_tool_calls, all_tool_results = [], []
        response = None
        _tool_retry_used = False

        for _ in range(max_iterations):
            response = await self.llm.ainvoke(messages, tools=tools if tools else None)
            if not self.llm.has_tool_calls(response):
                # Anti-hallucination: enforce tool usage


                if (tool_call_required and tools and not all_tool_calls
                        and not _tool_retry_used):
                    _tool_retry_used = True
                    print(f"\n️  [{self.name}] No tool calls — retrying with tool-use directive...")
                    messages.append({
                        "role": "user",
                        "content": (
                            "⚠️ CRITICAL: You MUST call the available tools before answering. "
                            "You have NO access to file contents, database records, or real data "
                            "without calling tools. If you answer without calling tools, your "
                            "response will be DISCARDED. Call a tool NOW."
                        ),
                    })
                    continue
                break
            tool_calls = self.llm.get_tool_calls(response)
            # Convert AIMessage to dict — LangChain objects break _to_langchain on next iteration


            messages.append({
                "role": "assistant",
                "content": response.content if hasattr(response, 'content') and response.content else "",
                "tool_calls": tool_calls,
            })
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]
                all_tool_calls.append(tc)
                print(f"\n [{self.name}] Calling: {tool_name}")
                try:
                    if hasattr(self, '_handle_local_tool'):
                        result_str = await self._handle_local_tool(tool_name, tool_args)
                    else:
                        result_str = f"[ERROR] No tool handler for: {tool_name}"
                except Exception as e:
                    result_str = f"[ERROR] {e}"
                all_tool_results.append({"tool": tool_name, "result": result_str})
                messages.append(self.llm.create_tool_message(tool_id, result_str))

        final = response.content if response and hasattr(response, 'content') else ""
        return {
            "response": final, "tool_calls": all_tool_calls,
            "tool_results": all_tool_results, "iterations": len(all_tool_calls),
            "tool_retry_used": _tool_retry_used,
        }

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
