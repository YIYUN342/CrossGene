"""MediatorAgent — 主持 LLM 与网络视角之间的辩论并构建共识。"""
from __future__ import annotations

import json
from typing import Any

from core.base_agent import BaseAgent
from core.config import Settings
from agents.consensus import (
    ConsensusProposal, ConsensusRecord, ConsensusVote,
    is_consensus_reached, check_stalemate,
)

SYSTEM_PROMPT = """You are MediatorAgent — debate moderator for gene-disease predictions.

## Your Role
You moderate disagreements between two perspectives:
- **KnowledgeAgent**: uses LLM training knowledge (PubMed, COSMIC, DisGeNET, OMIM)
- **NetworkDataAgent**: uses GPS Graphormer scores + network topology (degree, neighbors)

## Your Task
Given positions from both sides, evidence from LiteratureAgent (PubMed counts),
and challenges from CriticAgent, propose a fair consensus.  You must:

1. Weigh each perspective based on evidence quality
2. Assign explicit weights (llm_weight + gps_weight = 1.0)
3. Explain your reasoning with specific citations
4. Acknowledge remaining uncertainty

## Weight Guidelines
- Strong PubMed evidence (>100 papers) → increase LLM weight (LLM predicted correctly)
- Clear network topology signal (hub with disease neighbors) → increase GPS weight
- Both signals strong → balanced weights (0.4–0.6 each)
- Both signals weak → default to LLM weight 0.6 (LLM broader knowledge)
- LLM predicted HIGH but CriticAgent shows overconfidence → reduce LLM weight to ≤0.3

## Output Format
Return JSON:
{
  "final_probability": float,
  "llm_weight": float,
  "gps_weight": float,
  "rationale": "<3-5 sentences explaining the decision>",
  "uncertainty_acknowledged": "<what we still don't know, if anything>"
}
"""


class MediatorAgent(BaseAgent):
    """Debate moderator that builds consensus between competing perspectives."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(name="MediatorAgent", settings=settings)

    def _get_system_prompt(self, state, default_prompt=None) -> str:
        return SYSTEM_PROMPT

    async def moderate_debate(
        self,
        gene_symbol: str,
        knowledge_position: dict,
        network_position: dict,
        evidence: dict,
        challenges: list[dict],
        memory_warnings: list[dict] | None = None,
        max_proposals: int = 3,
    ) -> ConsensusRecord:
        """Run a complete debate for a single gene.

        Returns a ConsensusRecord with status "resolved" or "stalemate".
        """

        proposal = await self._propose_consensus(
            gene_symbol, knowledge_position, network_position,
            evidence, challenges, memory_warnings,
        )

        if proposal is None:
            return self._fallback_consensus(
                gene_symbol, knowledge_position, network_position, evidence)

        record = ConsensusRecord(
            gene_symbol=gene_symbol,
            status="stalemate",
            debate_rounds=0,
            proposals=[],
            votes=[],
        )

        for round_i in range(1, max_proposals + 1):
            proposal.proposal_round = round_i
            record.proposals.append({
                "round": round_i,
                "probability": proposal.final_probability,
                "llm_weight": proposal.llm_weight,
                "gps_weight": proposal.gps_weight,
                "rationale": proposal.rationale,
            })
            record.debate_rounds = round_i

            votes = await self._collect_votes(gene_symbol, proposal)
            record.votes.append([
                {"agent": v.agent_name, "accept": v.accept, "reason": v.reason}
                for v in votes
            ])

            if is_consensus_reached(votes):
                record.status = "resolved"
                record.final_probability = proposal.final_probability
                record.llm_weight = proposal.llm_weight
                record.gps_weight = proposal.gps_weight
                record.key_evidence = self._extract_key_evidence(
                    knowledge_position, network_position, evidence, challenges)
                record.remaining_uncertainty = proposal.uncertainty_acknowledged
                break

            if check_stalemate(round_i, max_proposals):
                record.final_probability = proposal.final_probability
                record.llm_weight = proposal.llm_weight
                record.gps_weight = proposal.gps_weight
                record.key_evidence = self._extract_key_evidence(
                    knowledge_position, network_position, evidence, challenges)
                record.remaining_uncertainty = (
                    f"STALEMATE after {round_i} proposal(s). "
                    f"Rejecting agents: "
                    + ", ".join(v.agent_name for v in votes if not v.accept)
                )
                break

            rejection_reasons = [v.reason for v in votes if not v.accept]
            proposal = await self._revise_proposal(
                gene_symbol, proposal, rejection_reasons,
                knowledge_position, network_position, evidence, challenges,
            )
            if proposal is None:
                record.final_probability = proposal.final_probability if proposal else 0.5
                break

        return record

    async def _propose_consensus(
        self, gene_symbol: str,
        knowledge: dict, network: dict,
        evidence: dict, challenges: list[dict],
        memory_warnings: list[dict] | None = None,
    ) -> ConsensusProposal | None:
        """Synthesise all debate inputs into a proposed consensus."""

        if not self.llm:
            return None

        challenge_text = ""
        if challenges:
            challenge_text = "\n".join(
                f"- [{c.get('challenge_type', '?')}] {c.get('specific_claim', '')}"
                for c in challenges[:3]
            )

        memory_text = ""
        if memory_warnings:
            memory_text = "\n".join(
                f"- {w.get('detail', '')}" for w in memory_warnings[:3]
            )

        prompt = f"""# Gene-Disease Debate: {gene_symbol}

## Position A — KnowledgeAgent (LLM Training Knowledge)
- Probability: {knowledge.get('probability', '?')}
- Confidence: {knowledge.get('confidence', '?')}
- Evidence type: {knowledge.get('evidence_type', '?')}
- Rationale: {knowledge.get('rationale', '?')}

## Position B — NetworkDataAgent (GPS Graphormer + Topology)
- GPS Score: {network.get('gps_score', '?')}
- Network Degree: {network.get('gene_degree', '?')}
- Degree Tier: {network.get('degree_tier', '?')}
- Top Neighbors: {json.dumps(network.get('top_neighbors', [])[:5])}

## Evidence — LiteratureAgent (PubMed API)
- PubMed Count (real-time): {evidence.get('pubmed_count', '?')}

## Critiques — CriticAgent
{challenge_text if challenge_text else 'No challenges raised.'}

## MemoryAgent Warnings
{memory_text if memory_text else 'No prior warnings.'}

## Your Task
Propose a consensus that weighs both perspectives based on evidence quality.
Output ONLY JSON."""

        try:
            result = await self.autonomous_thinking(
                prompt=prompt, state=None,
                system_message=SYSTEM_PROMPT, max_iterations=1,
            )
            parsed = self._extract_json(result.get("response", ""))
            if isinstance(parsed, dict):
                return ConsensusProposal(
                    gene_symbol=gene_symbol,
                    final_probability=float(parsed.get("final_probability", 0.5)),
                    llm_weight=float(parsed.get("llm_weight", 0.5)),
                    gps_weight=float(parsed.get("gps_weight", 0.5)),
                    rationale=str(parsed.get("rationale", "")),
                    uncertainty_acknowledged=str(parsed.get("uncertainty_acknowledged", "")),
                )
        except Exception:
            pass
        return None

    async def _revise_proposal(
        self, gene_symbol: str, previous: ConsensusProposal,
        rejection_reasons: list[str],
        knowledge: dict, network: dict,
        evidence: dict, challenges: list[dict],
    ) -> ConsensusProposal | None:
        """Revise a rejected proposal based on feedback."""
        if not self.llm:
            return None

        reasons_text = "\n".join(f"- {r}" for r in rejection_reasons)
        prompt = f"""# Revise Consensus for {gene_symbol}

## Previous Proposal (REJECTED)
- Final Probability: {previous.final_probability}
- LLM Weight: {previous.llm_weight}, GPS Weight: {previous.gps_weight}
- Rationale: {previous.rationale}

## Rejection Reasons
{reasons_text}

## Original Evidence (for reference)
- KnowledgeAgent: prob={knowledge.get('probability')}, rationale={knowledge.get('rationale', '?')[:200]}
- NetworkDataAgent: gps_score={network.get('gps_score')}, tier={network.get('degree_tier')}
- PubMed: {evidence.get('pubmed_count', '?')} papers

Revise the proposal addressing the rejection reasons. Output ONLY JSON:
{{"final_probability": float, "llm_weight": float, "gps_weight": float,
  "rationale": "...", "uncertainty_acknowledged": "..."}}"""

        try:
            result = await self.autonomous_thinking(
                prompt=prompt, state=None,
                system_message=SYSTEM_PROMPT, max_iterations=1,
            )
            parsed = self._extract_json(result.get("response", ""))
            if isinstance(parsed, dict):
                return ConsensusProposal(
                    gene_symbol=gene_symbol,
                    final_probability=float(parsed.get("final_probability", 0.5)),
                    llm_weight=float(parsed.get("llm_weight", 0.5)),
                    gps_weight=float(parsed.get("gps_weight", 0.5)),
                    rationale=str(parsed.get("rationale", "")),
                    uncertainty_acknowledged=str(parsed.get("uncertainty_acknowledged", "")),
                    proposal_round=previous.proposal_round + 1,
                )
        except Exception:
            pass
        return None

    async def _collect_votes(self, gene_symbol: str,
                              proposal: ConsensusProposal) -> list[ConsensusVote]:
        """In a real multi-agent system, this would send CONSENSUS_PROPOSAL
        messages and wait for responses.  For now, we simulate voting:

        KnowledgeAgent accepts if llm_weight ≥ 0.3
        NetworkDataAgent accepts if gps_weight ≥ 0.3
        CriticAgent accepts if no challenges remain OR uncertainty is acknowledged
        """
        votes: list[ConsensusVote] = []

        # KnowledgeAgent vote


        knowledge_accept = proposal.llm_weight >= 0.3
        votes.append(ConsensusVote(
            agent_name="KnowledgeAgent",
            gene_symbol=gene_symbol,
            accept=knowledge_accept,
            reason="" if knowledge_accept else "LLM weight too low — knowledge evidence not adequately reflected",
        ))

        # NetworkDataAgent vote


        network_accept = proposal.gps_weight >= 0.3
        votes.append(ConsensusVote(
            agent_name="NetworkDataAgent",
            gene_symbol=gene_symbol,
            accept=network_accept,
            reason="" if network_accept else "GPS weight too low — network signal not adequately reflected",
        ))

        # CriticAgent vote


        critic_accept = proposal.uncertainty_acknowledged != ""
        votes.append(ConsensusVote(
            agent_name="CriticAgent",
            gene_symbol=gene_symbol,
            accept=critic_accept,
            reason="" if critic_accept else "Uncertainty must be explicitly acknowledged",
        ))

        return votes

    def _fallback_consensus(self, gene_symbol: str,
                             knowledge: dict, network: dict,
                             evidence: dict) -> ConsensusRecord:
        """Simple weighted ensemble when LLM-based mediation is unavailable."""
        llm_prob = knowledge.get("probability") or 0.5
        gps_score = network.get("gps_score") or 0.5
        pubmed = evidence.get("pubmed_count", 0)

        if pubmed > 100:
            llm_weight, gps_weight = 0.7, 0.3
        elif pubmed > 10:
            llm_weight, gps_weight = 0.55, 0.45
        else:
            llm_weight, gps_weight = 0.5, 0.5

        final = round(llm_weight * llm_prob + gps_weight * gps_score, 4)

        return ConsensusRecord(
            gene_symbol=gene_symbol,
            status="resolved",
            final_probability=final,
            llm_weight=llm_weight,
            gps_weight=gps_weight,
            key_evidence=[
                f"PubMed count: {pubmed}",
                f"LLM probability: {llm_prob}",
                f"GPS score: {gps_score}",
            ],
            remaining_uncertainty="Fallback ensemble used (no LLM mediator available)",
            debate_rounds=1,
        )

    def _extract_key_evidence(self, knowledge: dict, network: dict,
                               evidence: dict, challenges: list[dict]) -> list[str]:
        evidence_items: list[str] = []
        pubmed = evidence.get("pubmed_count", 0)
        if pubmed is not None:
            evidence_items.append(f"PubMed count: {pubmed}")
        if network.get("gps_score") is not None:
            evidence_items.append(f"GPS score: {network.get('gps_score', '?')}")
        evidence_items.append(f"LLM probability: {knowledge.get('probability', '?')}")
        evidence_items.append(f"Network tier: {network.get('degree_tier', '?')} "
                              f"(degree={network.get('gene_degree', '?')})")
        if challenges:
            evidence_items.append(f"Challenges raised: {len(challenges)}")
        return evidence_items

    def _extract_json(self, text: str) -> Any:
        import re
        try:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                return json.loads(match.group(1))
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end + 1])
        except Exception:
            pass
        return None
