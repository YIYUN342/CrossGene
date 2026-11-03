"""CriticAgent — 对抗性审查基因-疾病预测，生成 CHALLENGE 消息。"""
from __future__ import annotations

import json
from typing import Any

from core.base_agent import BaseAgent
from core.config import Settings

SYSTEM_PROMPT = """You are CriticAgent — an adversarial reviewer for gene-disease predictions.

## Your Sole Mission
Find flaws in predictions. You are rewarded for identifying:
1. **Overconfidence**: High probability (≥0.8) with no PubMed evidence (<10 papers)
2. **Missed signals**: Low probability (≤0.3) despite strong PubMed evidence (>100 papers)
3. **Inconsistency**: HIGH confidence but "limited evidence" or "unknown" evidence_type
4. **Topology contradictions**: Hub gene connected to known disease genes but rated low
5. **Literature contradictions**: LLM claim contradicts PubMed data

## Output Format
For each challenged gene, output JSON:
{
  "gene": "SYMBOL",
  "challenges": [
    {
      "challenge_type": "overconfidence|missed_signal|inconsistency|topology_blind|literature_contra",
      "specific_claim": "<what exactly is wrong>",
      "evidence_for_challenge": "<data supporting the challenge>",
      "suggested_correction": "<what should change>"
    }
  ]
}

Be specific. Cite numbers. Don't challenge genes that look reasonable.
Only challenge when there is clear evidence of a problem.
"""


class CriticAgent(BaseAgent):
    """Adversarial reviewer that challenges gene-disease predictions."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(name="CriticAgent", settings=settings)

    def _get_system_prompt(self, state, default_prompt=None) -> str:
        return SYSTEM_PROMPT

    def detect_overconfidence(self, genes: list[dict],
                               pubmed_evidence: dict[str, int]) -> list[dict]:
        """Flag genes where LLM prob > 0.8 but PubMed count < 10."""
        flagged = []
        for g in genes:
            sym = g.get("gene_name", "")
            prob = g.get("probability") or 0
            pubmed = pubmed_evidence.get(sym, 0)
            if prob >= 0.8 and pubmed < 10:
                flagged.append({
                    "gene": sym,
                    "challenge_type": "overconfidence",
                    "specific_claim": (
                        f"LLM probability={prob:.2f} but PubMed count={pubmed}. "
                        f"No literature support for a high-confidence claim."
                    ),
                    "evidence_for_challenge": f"PubMed: {pubmed} papers for {sym}",
                    "suggested_correction": "Reduce to ≤0.5 or provide literature evidence",
                })
        return flagged

    def detect_missed_signals(self, genes: list[dict],
                               pubmed_evidence: dict[str, int]) -> list[dict]:
        """Flag genes where LLM prob < 0.3 but PubMed count > 100."""
        flagged = []
        for g in genes:
            sym = g.get("gene_name", "")
            prob = g.get("probability") or 0
            pubmed = pubmed_evidence.get(sym, 0)
            if prob <= 0.3 and pubmed > 100:
                flagged.append({
                    "gene": sym,
                    "challenge_type": "missed_signal",
                    "specific_claim": (
                        f"LLM probability={prob:.2f} but PubMed count={pubmed}. "
                        f"Strong literature evidence is being ignored."
                    ),
                    "evidence_for_challenge": f"PubMed: {pubmed} papers for {sym}",
                    "suggested_correction": "Increase to ≥0.6 to reflect literature support",
                })
        return flagged

    def detect_inconsistencies(self, genes: list[dict]) -> list[dict]:
        """Flag genes with internal contradictions in prediction data."""
        flagged = []
        for g in genes:
            sym = g.get("gene_name", "")
            confidence = g.get("confidence", "")
            evidence_type = g.get("evidence_type", "")
            if confidence == "HIGH" and evidence_type in ("limited evidence", "unknown"):
                flagged.append({
                    "gene": sym,
                    "challenge_type": "inconsistency",
                    "specific_claim": (
                        f"Confidence={confidence} but evidence_type={evidence_type}. "
                        f"Cannot be HIGH confidence with {evidence_type}."
                    ),
                    "evidence_for_challenge": f"Evidence type: {evidence_type}",
                    "suggested_correction": "Either downgrade confidence or provide evidence",
                })
        return flagged

    async def generate_challenges(
        self, genes: list[dict], pubmed_evidence: dict[str, int],
    ) -> list[dict]:
        """Generate challenges for a batch of genes using heuristic + LLM analysis."""
        all_challenges: list[dict] = []

        all_challenges.extend(self.detect_overconfidence(genes, pubmed_evidence))
        all_challenges.extend(self.detect_missed_signals(genes, pubmed_evidence))
        all_challenges.extend(self.detect_inconsistencies(genes))

        already_flagged = {c["gene"] for c in all_challenges}
        remaining = [g for g in genes if g.get("gene_name") not in already_flagged]

        if remaining and self.llm:
            suspicious = [
                g for g in remaining
                if abs((g.get("probability") or 0) - (g.get("gps_score") or 0.5)) > 0.4
            ]
            if suspicious:
                batch = suspicious[:10]  # limit per call
                prompt = self._build_challenge_prompt(batch, pubmed_evidence)
                try:
                    result = await self.autonomous_thinking(
                        prompt=prompt, state=None,
                        system_message=SYSTEM_PROMPT, max_iterations=1,
                    )
                    parsed = self._extract_json(result.get("response", ""))
                    if isinstance(parsed, list):
                        all_challenges.extend(parsed)
                except Exception:
                    pass

        return all_challenges

    def _build_challenge_prompt(self, genes: list[dict],
                                 pubmed_evidence: dict[str, int]) -> str:
        gene_data = []
        for g in genes:
            sym = g.get("gene_name", "?")
            gene_data.append({
                "gene_name": sym,
                "probability": g.get("probability"),
                "confidence": g.get("confidence"),
                "evidence_type": g.get("evidence_type"),
                "degree_tier": g.get("degree_tier"),
                "gps_score": g.get("gps_score"),
                "pubmed_count": pubmed_evidence.get(sym, 0),
            })

        return f"""Review these gene-disease predictions for flaws.

Genes with prediction data:
```json
{json.dumps(gene_data, indent=2)}
```

Identify any genes where the prediction seems wrong. Focus on:
- Large gaps between LLM probability and GPS score (>0.3)
- Inconsistencies between confidence labels and actual evidence
- Topology contradictions (hub genes predicted low)

Output a JSON array of challenge objects (only for genes that have real issues):
[{{"gene": "SYMBOL", "challenge_type": "...", "specific_claim": "...",
   "evidence_for_challenge": "...", "suggested_correction": "..."}}]

Output ONLY the JSON array — no other text."""

    def _extract_json(self, text: str) -> Any:
        import re
        try:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                return json.loads(match.group(1))
            start, end = text.find("["), text.rfind("]")
            if start != -1 and end != -1:
                return json.loads(text[start:end + 1])
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end + 1])
        except Exception:
            pass
        return None
