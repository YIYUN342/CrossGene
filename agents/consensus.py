"""GeneAgent 辩论协议的共识数据结构 (Proposal / Vote / Record)。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConsensusProposal:
    """A proposed resolution for a disputed gene, created by MediatorAgent."""
    gene_symbol: str
    final_probability: float
    llm_weight: float          # 0.0–1.0, how much to trust LLM knowledge
    gps_weight: float         # 1.0 – llm_weight, how much to trust GPS/network
    rationale: str             # explains the weight assignment
    uncertainty_acknowledged: str = ""
    proposal_round: int = 1    # which proposal attempt (1, 2, 3)
    debate_id: str = ""


@dataclass
class ConsensusVote:
    """An agent's vote on a ConsensusProposal."""
    agent_name: str
    gene_symbol: str
    accept: bool
    reason: str = ""           # required if accept=False
    proposal_id: str = ""


@dataclass
class ConsensusRecord:
    """Final outcome for a disputed gene, including full debate history."""
    gene_symbol: str
    status: str                # "resolved" | "stalemate"
    final_probability: float | None = None
    llm_weight: float | None = None
    gps_weight: float | None = None
    key_evidence: list[str] = field(default_factory=list)
    remaining_uncertainty: str = ""
    debate_rounds: int = 0
    proposals: list[dict[str, Any]] = field(default_factory=list)
    votes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gene": self.gene_symbol,
            "status": self.status,
            "final_probability": self.final_probability,
            "llm_weight": self.llm_weight,
            "gps_weight": self.gps_weight,
            "key_evidence": self.key_evidence,
            "remaining_uncertainty": self.remaining_uncertainty,
            "debate_rounds": self.debate_rounds,
            "proposals": self.proposals,
            "votes": self.votes,
        }


def is_consensus_reached(votes: list[ConsensusVote]) -> bool:
    """True when ALL votes accept the proposal."""
    return all(v.accept for v in votes) if votes else False


def check_stalemate(proposal_round: int, max_rounds: int = 3,
                    rejections: list[ConsensusVote] | None = None) -> bool:
    """True when max re-proposals have been exhausted without consensus."""
    return proposal_round >= max_rounds
