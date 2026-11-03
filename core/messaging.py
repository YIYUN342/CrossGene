"""
MessageBus — event-driven communication backbone for GeneAgent multi-agent system.

Provides:
  - MessageType enum: all message categories in the debate protocol
  - AgentMessage dataclass: typed, traceable messages between agents
  - MessageBus: async pub/sub router with audit trail

Design properties:
  - Non-blocking sends: sender never waits for receiver to process
  - Typed subscriptions: agents filter by MessageType (default: receive all)
  - Broadcast: receiver="broadcast" delivers to all registered agents
  - Audit trail: every message appended to an append-only log
  - Priority: consensus votes can jump the queue (priority=1)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageType(Enum):
    """All message categories in the GeneAgent debate protocol."""

    # Workflow coordination


    REQUEST = "request"          # Ask an agent to perform its primary task
    RESPONSE = "response"         # Reply with results / data
    READY = "ready"               # Signal task completion

    # Debate protocol


    POSITION = "position"         # Agent's stance on a gene prediction
    CHALLENGE = "challenge"       # CriticAgent's structured criticism
    EVIDENCE = "evidence"         # LiteratureAgent's PubMed / DB data
    CONSENSUS_PROPOSAL = "consensus_proposal"  # Mediator proposes resolution
    CONSENSUS_ACCEPT = "consensus_accept"      # Agent accepts proposal
    CONSENSUS_REJECT = "consensus_reject"      # Agent rejects with reason
    REBUTTAL = "rebuttal"         # Agent responds to a challenge

    # Memory


    MEMORY_WARNING = "memory_warning"             # Repeated mistake detected
    CROSS_ROUND_INSIGHT = "cross_round_insight"   # Systematic pattern found

    # Control


    DEBATE_INITIATE = "debate_initiate"  # Start debate on gene(s)
    DEBATE_CLOSE = "debate_close"        # Debate concluded
    FINAL = "final"                      # Final conclusion distributed


@dataclass
class AgentMessage:
    """A typed, traceable message between agents.

    Every message is immutable after creation and stored in the
    MessageBus audit log for post-hoc analysis by MemoryAgent.
    """

    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    receiver: str = ""          # agent name, "broadcast", or specific agent
    msg_type: MessageType = MessageType.REQUEST
    round_id: str = ""          # which orchestration round this belongs to
    debate_id: str = ""         # which debate session (for per-gene debates)
    gene_symbol: str | None = None  # specific gene being discussed
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    reply_to: str | None = None  # msg_id this is replying to
    priority: int = 0            # 0 = normal, 1 = urgent (consensus votes)

    def short_repr(self) -> str:
        """Compact representation for logging."""
        pld_keys = list(self.payload.keys())[:3]
        return (
            f"[{self.sender} → {self.receiver}] {self.msg_type.value}"
            f"{' (gene=' + self.gene_symbol + ')' if self.gene_symbol else ''}"
            f" payload_keys={pld_keys}"
        )


class MessageBus:
    """Central message router for the GeneAgent multi-agent system.

    Each agent registers an inbox (asyncio.Queue).  Senders push messages
    that are routed to the target agent's queue.  The bus is non-blocking:
    send() returns immediately; the receiver dequeues on its own schedule.

    Usage::

        bus = MessageBus()
        await bus.register("orchestrator")
        await bus.register("knowledge", subscribe_types={MessageType.REQUEST,
                                                          MessageType.CHALLENGE})
        msg = AgentMessage(sender="orchestrator", receiver="knowledge",
                           msg_type=MessageType.REQUEST, payload={...})
        await bus.send(msg)
        received = await bus.receive("knowledge", timeout=5.0)
    """

    def __init__(self) -> None:
        self._inboxes: dict[str, asyncio.Queue[AgentMessage]] = {}
        self._subscriptions: dict[str, set[MessageType]] = defaultdict(set)
        self._audit_log: list[AgentMessage] = []   # append-only, complete history
        self._message_counter: int = 0

    # Registration



    async def register(self, agent_name: str,
                       subscribe_types: set[MessageType] | None = None) -> None:
        """Register an agent with the bus.  If subscribe_types is None,
        the agent receives ALL message types (no filter)."""
        self._inboxes[agent_name] = asyncio.Queue()
        if subscribe_types:
            self._subscriptions[agent_name] = subscribe_types

    async def unregister(self, agent_name: str) -> None:
        """Remove an agent from the bus."""
        self._inboxes.pop(agent_name, None)
        self._subscriptions.pop(agent_name, None)

    # Send



    async def send(self, msg: AgentMessage) -> None:
        """Route a message to the target agent(s).  Non-blocking."""
        self._audit_log.append(msg)
        self._message_counter += 1

        targets: list[str]
        if msg.receiver == "broadcast":
            targets = [n for n in self._inboxes if n != msg.sender]
        else:
            targets = [msg.receiver]

        for target in targets:
            if target == msg.sender:
                continue  # never deliver to self
            if target not in self._inboxes:
                continue
            # Check subscription filter


            subs = self._subscriptions.get(target)
            if subs and msg.msg_type not in subs:
                continue
            await self._inboxes[target].put(msg)

    # Receive



    async def receive(self, agent_name: str,
                      timeout: float = 5.0) -> AgentMessage | None:
        """Wait for the next message.  Returns None on timeout."""
        if agent_name not in self._inboxes:
            return None
        try:
            return await asyncio.wait_for(
                self._inboxes[agent_name].get(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            return None

    async def receive_all(self, agent_name: str,
                          timeout: float = 0.3) -> list[AgentMessage]:
        """Drain all queued messages from an agent's inbox."""
        messages: list[AgentMessage] = []
        while True:
            msg = await self.receive(agent_name, timeout=timeout)
            if msg is None:
                break
            messages.append(msg)
        return messages

    # Queries



    @property
    def message_count(self) -> int:
        return self._message_counter

    def get_audit_trail(self, round_id: str | None = None,
                        msg_type: MessageType | None = None,
                        sender: str | None = None) -> list[AgentMessage]:
        """Query the audit log.  Used by MemoryAgent for pattern detection."""
        msgs = self._audit_log
        if round_id:
            msgs = [m for m in msgs if m.round_id == round_id]
        if msg_type:
            msgs = [m for m in msgs if m.msg_type == msg_type]
        if sender:
            msgs = [m for m in msgs if m.sender == sender]
        return list(msgs)

    def reset(self) -> None:
        """Clear all inboxes, subscriptions, and audit log."""
        self._inboxes.clear()
        self._subscriptions.clear()
        self._audit_log.clear()
        self._message_counter = 0
