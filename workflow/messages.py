"""
Message factory helpers for the GeneAgent debate protocol.

Lightweight convenience functions that build properly-typed AgentMessage
payloads.  Re-exports core.messaging types so callers only need one import.
"""
from __future__ import annotations

from core.messaging import AgentMessage, MessageBus, MessageType  # noqa: F401


# Workflow coordination



def make_request(task: str, context: dict | None = None,
                 round_id: str = "") -> dict:
    """Build a REQUEST payload asking an agent to perform its primary task."""
    return {"task": task, "context": context or {}, "round_id": round_id}


def make_response(data: dict, status: str = "success") -> dict:
    """Build a RESPONSE payload with results."""
    return {"status": status, "data": data}


def make_ready(summary: str = "") -> dict:
    """Build a READY payload signalling task completion."""
    return {"summary": summary}


# Debate protocol



def make_position(gene_symbol: str, probability: float,
                  confidence: str, rationale: str,
                  source: str = "llm_knowledge") -> dict:
    """Build a POSITION payload for a gene prediction stance."""
    return {
        "gene_symbol": gene_symbol,
        "probability": probability,
        "confidence": confidence,
        "rationale": rationale,
        "source": source,
    }


def make_challenge(gene_symbol: str, challenge_type: str,
                   specific_claim: str, evidence_for_challenge: str,
                   suggested_correction: str) -> dict:
    """Build a CHALLENGE payload from CriticAgent."""
    return {
        "gene_symbol": gene_symbol,
        "challenge_type": challenge_type,
        "specific_claim": specific_claim,
        "evidence_for_challenge": evidence_for_challenge,
        "suggested_correction": suggested_correction,
    }


def make_evidence(gene_symbol: str, pubmed_count: int | None = None,
                  mygene_info: dict | None = None,
                  open_targets: list | None = None) -> dict:
    """Build an EVIDENCE payload from LiteratureAgent."""
    return {
        "gene_symbol": gene_symbol,
        "pubmed_count": pubmed_count,
        "mygene": mygene_info or {},
        "open_targets": open_targets or [],
    }


def make_consensus_proposal(gene_symbol: str, final_probability: float,
                            llm_weight: float, gps_weight: float,
                            rationale: str,
                            uncertainty: str = "") -> dict:
    """Build a CONSENSUS_PROPOSAL payload from MediatorAgent."""
    return {
        "gene_symbol": gene_symbol,
        "final_probability": final_probability,
        "llm_weight": llm_weight,
        "gps_weight": gps_weight,
        "rationale": rationale,
        "uncertainty_acknowledged": uncertainty,
    }


def make_consensus_vote(gene_symbol: str, accept: bool,
                        reason: str = "") -> dict:
    """Build a CONSENSUS_ACCEPT or CONSENSUS_REJECT payload."""
    return {
        "gene_symbol": gene_symbol,
        "accept": accept,
        "reason": reason,
    }


# Memory



def make_memory_warning(gene_symbol: str, warning_type: str,
                        detail: str) -> dict:
    """Build a MEMORY_WARNING payload."""
    return {
        "gene_symbol": gene_symbol,
        "warning_type": warning_type,
        "detail": detail,
    }


def make_cross_round_insight(pattern: str, affected_genes: list[str],
                             recommendation: str) -> dict:
    """Build a CROSS_ROUND_INSIGHT payload."""
    return {
        "pattern": pattern,
        "affected_genes": affected_genes,
        "recommendation": recommendation,
    }
