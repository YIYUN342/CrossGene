"""
Debate session management for GeneAgent.

DebateSession: orchestrates one gene's complete debate lifecycle
debate_required(): predicate — does this gene need a debate?
extract_disputed_genes(): identify which genes have LLM-GPS disagreement
"""
from __future__ import annotations

from typing import Any

from agents.consensus import ConsensusRecord


def debate_required(llm_prob: float | None, gps_score: float | None,
                    pubmed_count: int | None = None,
                    threshold: float = 0.3) -> bool:
    """Return True if a gene's predictions warrant a debate.

    A debate is triggered when:
      - LLM-GPS gap exceeds threshold (|llm - gps| > 0.3), OR
      - PubMed evidence contradicts LLM (high prob, low count, or vice versa)
    """
    if llm_prob is None or gps_score is None:
        return False

    # LLM-GPS disagreement


    gap = abs(llm_prob - gps_score)
    if gap > threshold:
        return True

    # PubMed-LM discordance


    if pubmed_count is not None:
        if llm_prob >= 0.8 and pubmed_count < 10:
            return True  # overconfident
        if llm_prob <= 0.3 and pubmed_count > 100:
            return True  # missed signal

    return False


def extract_disputed_genes(
    predictions: dict[str, dict[str, Any]],
    pubmed_evidence: dict[str, int] | None = None,
    threshold: float = 0.3,
) -> list[str]:
    """Return gene symbols that need debate.

    Candidates: genes with LLM-GPS gap > threshold OR PubMed discordance.
    Sorted by gap descending (most disputed first).
    """
    pubmed_evidence = pubmed_evidence or {}
    disputed: list[tuple[str, float]] = []

    for sym, pred in predictions.items():
        if not isinstance(pred, dict):
            continue
        llm_prob = pred.get("llm_prob") or pred.get("probability")
        gps_score = pred.get("gps_score")
        pubmed = pubmed_evidence.get(sym)

        if debate_required(llm_prob, gps_score, pubmed, threshold):
            gap = abs((llm_prob or 0.5) - (gps_score or 0.5))
            disputed.append((sym, gap))

    disputed.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in disputed]


class DebateSession:
    """Tracks the lifecycle of a single gene's debate.

    Not an agent — a lightweight state machine used by the Orchestrator
    to manage debate rounds, timeouts, and result collection.
    """

    def __init__(self, gene_symbol: str, round_id: str = "") -> None:
        self.gene_symbol = gene_symbol
        self.round_id = round_id
        self.debate_id = f"debate_{gene_symbol}_{round_id}"
        self.status: str = "pending"   # pending | debating | resolved | stalemate
        self.current_round: int = 0
        self.max_rounds: int = 3
        self.result: ConsensusRecord | None = None
        self._positions: dict[str, dict] = {}
        self._evidence: dict[str, Any] = {}
        self._challenges: list[dict] = []

    def add_position(self, agent_name: str, position: dict) -> None:
        self._positions[agent_name] = position

    def add_evidence(self, evidence: dict) -> None:
        self._evidence = evidence

    def add_challenge(self, challenge: dict) -> None:
        self._challenges.append(challenge)

    @property
    def knowledge_position(self) -> dict:
        return self._positions.get("KnowledgeAgent", {})

    @property
    def network_position(self) -> dict:
        return self._positions.get("NetworkDataAgent", {})

    @property
    def is_complete(self) -> bool:
        return self.status in ("resolved", "stalemate")

    def mark_resolved(self, record: ConsensusRecord) -> None:
        self.status = "resolved"
        self.result = record

    def mark_stalemate(self, record: ConsensusRecord) -> None:
        self.status = "stalemate"
        self.result = record
