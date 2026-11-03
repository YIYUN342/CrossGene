"""
Lightweight LLM client for GeneAgent. Supports OpenAI-compatible APIs (DeepSeek, GPT, etc.).
"""
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config):
        self.config = config
        self.chat = ChatOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens or None,
            timeout=120,
            max_retries=2,
        )

    def _to_langchain(self, messages: list[dict]) -> list:
        out = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                out.append(SystemMessage(content=content))
            elif role == "user":
                out.append(HumanMessage(content=content))
            elif role == "assistant":
                ai = AIMessage(content=content)
                if "tool_calls" in m:
                    ai.tool_calls = m["tool_calls"]
                out.append(ai)
            elif role == "tool":
                out.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id", "")))
        return out

    async def ainvoke(self, messages: list[dict], tools: list[dict] | None = None):
        lc_msgs = self._to_langchain(messages)
        kwargs = {}
        if tools:
            kwargs["tools"] = tools
        return await self.chat.ainvoke(lc_msgs, **kwargs)

    async def ainvoke_streaming(self, messages: list[dict], tools=None, print_stream=True):
        lc_msgs = self._to_langchain(messages)
        kwargs = {"streaming": True}
        if tools:
            kwargs["tools"] = tools

        collected = []
        async for chunk in self.chat.astream(lc_msgs, **kwargs):
            if chunk.content:
                collected.append(chunk.content)
                if print_stream:
                    print(chunk.content, end="", flush=True)
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                # Store tool calls from the final chunk


                pass

        if print_stream:
            print()

        # Build response from collected chunks


        full_content = "".join(collected)
        return AIMessage(content=full_content)

    async def ainvoke_simple(self, prompt: str, system_message: str | None = None) -> str:
        msgs = []
        if system_message:
            msgs.append({"role": "system", "content": system_message})
        msgs.append({"role": "user", "content": prompt})
        resp = await self.ainvoke(msgs)
        return resp.content if hasattr(resp, 'content') else str(resp)

    def has_tool_calls(self, response) -> bool:
        return hasattr(response, 'tool_calls') and response.tool_calls

    def get_tool_calls(self, response) -> list[dict]:
        result = []
        for tc in response.tool_calls:
            result.append({
                "id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
                "type": "function",
                "function": {"name": tc.get("name", ""), "arguments": tc.get("args", {})},
            })
        return result

    def create_tool_message(self, tool_call_id: str, content: str) -> dict:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}
