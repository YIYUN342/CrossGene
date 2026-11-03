"""MemoryAgent — 归档每轮结果并为 MetaAgent 提炼洞察。"""

import json
from typing import Any

from core.base_agent import BaseAgent
from core.config import Settings


SYSTEM_PROMPT = """You are MemoryAgent — Knowledge Keeper for GeneAgent.

## Role
Archive each iteration's gene-disease prediction results and extract actionable insights for the Chief Scientist (MetaAgent).

## What to Identify
1. **LLM-GPS Disagreements**: genes where LLM probability and GPS score diverge significantly (gap > 0.3)
2. **LLM Uncertainty**: genes with probability near 0.5 (the LLM doesn't know)
3. **PubMed Discordance**: genes where LLM prediction is HIGH but PubMed evidence is LOW (or vice versa)
4. **Improvement Trends**: did AUC/F1 improve from the previous round?
5. **Network Patterns**: are hub genes consistently predicted higher?

## Output Format
Return a concise JSON summary:
{
  "top_disagreements": [{"gene": "SYMBOL", "llm": 0.XX, "gps": 0.XX}],
  "uncertain_genes": ["SYMBOL1", "SYMBOL2"],
  "pubmed_discordant": [{"gene": "SYMBOL", "llm_prob": 0.XX, "pubmed_count": N}],
  "improvement": {"auc_delta": +0.XXX, "f1_delta": +0.XXX},
  "key_observation": "<1 sentence synthesis>",
  "recommendation": "<what MetaAgent should tell LLMFabAgent for next round>"
}
"""


class MemoryAgent(BaseAgent):
    """Archives iteration results and distills insights for MetaAgent."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(name="MemoryAgent", settings=settings)

    def _get_system_prompt(self, state, default_prompt=None):
        return SYSTEM_PROMPT

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        iteration = state.get("iteration", 0)
        fab_results = state.get("fab_results", {})
        analysis_report = state.get("analysis_report", "")
        literature_report = state.get("literature_report", "")
        iteration_history = state.get("iteration_history", [])

        print(f"\n[MemoryAgent] Archiving iteration {iteration}...")

        analysis_data = {}
        try:
            analysis_data = json.loads(analysis_report) if isinstance(analysis_report, str) else analysis_report
        except Exception:
            pass

        # 1. LLM-GPS disagreements


        proposed = fab_results.get("proposed_ensemble", {}) if isinstance(fab_results, dict) else {}
        disagreements = []
        for sym, pred in proposed.items():
            if not isinstance(pred, dict):
                continue
            llm = pred.get("llm_prob", 0) or 0
            gps = pred.get("gps_score", 0) or 0
            if abs(llm - gps) > 0.3:
                disagreements.append({"gene": sym, "llm": round(llm, 3), "gps": round(gps, 3)})

        # 2. Uncertain genes (prob ≈ 0.5)


        baseline = fab_results.get("baseline_llm_only", {}) if isinstance(fab_results, dict) else {}
        uncertain = []
        for sym, pred in baseline.items():
            if not isinstance(pred, dict):
                continue
            prob = pred.get("probability", 0) or 0
            if 0.4 <= prob <= 0.6:
                uncertain.append(sym)

        # 3. PubMed discordance


        per_gene = analysis_data.get("per_gene_analysis", [])
        pubmed_discordant = []
        for entry in per_gene:
            align = entry.get("alignment", "")
            if align in ("llm_high_db_low", "llm_low_db_high"):
                pubmed_discordant.append({
                    "gene": entry.get("gene_name", "?"),
                    "llm_prob": entry.get("probability", 0),
                    "pubmed_count": entry.get("pubmed_count", 0),
                })

        # 4. Improvement from previous round


        ev = analysis_data.get("evaluation", {})
        proposed_ev = ev.get("proposed_ensemble", {})
        current_auc = proposed_ev.get("auc", 0) or 0
        current_f1 = proposed_ev.get("f1", 0) or 0

        prev_auc, prev_f1 = 0, 0
        if len(iteration_history) >= 1:
            prev_entry = iteration_history[-1]
            prev_ev = prev_entry.get("evaluation", {})
            prev_prop = prev_ev.get("proposed_ensemble", {})
            prev_auc = prev_prop.get("auc", 0) or 0
            prev_f1 = prev_prop.get("f1", 0) or 0

        auc_delta = round(current_auc - prev_auc, 4)
        f1_delta = round(current_f1 - prev_f1, 4)

        # 5. Network pattern


        topo = analysis_data.get("topology_analysis", {})
        deg_corr = topo.get("degree_probability_correlation", None)

        # 6. LLM synthesis


        if self.llm:
            insights = await self._llm_synthesize(
                disagreements[:10], uncertain[:10], pubmed_discordant[:10],
                auc_delta, f1_delta, deg_corr, iteration,
            )
        else:
            insights = {
                "top_disagreements": disagreements[:5],
                "uncertain_genes": uncertain[:10],
                "pubmed_discordant": pubmed_discordant[:5],
                "improvement": {"auc_delta": auc_delta, "f1_delta": f1_delta},
                "key_observation": f"Iteration {iteration}: AUC={current_auc:.4f}, "
                                   f"{len(disagreements)} LLM-GPS disagreements, "
                                   f"{len(uncertain)} uncertain genes.",
                "recommendation": f"Focus on {len(uncertain)} uncertain genes next round.",
            }

        memory_text = json.dumps(insights, indent=2, ensure_ascii=False)
        print(f"   Disagreements (LLM vs GPS): {len(disagreements)}")
        print(f"   Uncertain genes (prob ≈ 0.5): {len(uncertain)}")
        print(f"   PubMed discordant: {len(pubmed_discordant)}")
        print(f"   AUC delta: {auc_delta:+.4f}")

        round_summary = {
            "iteration": iteration,
            "round_label": fab_results.get("round_label", f"R{iteration}") if isinstance(fab_results, dict) else f"R{iteration}",
            "evaluation_auc": current_auc,
            "evaluation_f1": current_f1,
            "n_llm_predicted": sum(1 for p in baseline.values() if isinstance(p, dict) and p.get("probability") is not None),
            "n_ensemble": sum(1 for p in proposed.values() if isinstance(p, dict) and p.get("probability") is not None),
            "n_disagreements": len(disagreements),
            "n_uncertain": len(uncertain),
            "n_pubmed_discordant": len(pubmed_discordant),
        }
        all_round_results = state.get("all_round_results", [])
        all_round_results.append(round_summary)

        return {
            "memory_insights": memory_text,
            "all_round_results": all_round_results,
        }

    async def _llm_synthesize(self, disagreements, uncertain, pubmed_discordant,
                              auc_delta, f1_delta, deg_corr, iteration):
        """Use LLM to synthesize insights from structured data."""
        prompt = f"""# Iteration {iteration} Summary

## LLM-GPS Disagreements (gap > 0.3)
{json.dumps(disagreements[:10], indent=2)}

## Uncertain Genes (LLM prob between 0.4-0.6)
{json.dumps(uncertain[:10])}

## PubMed Discordant Genes
{json.dumps(pubmed_discordant[:10], indent=2)}

## Improvement
- AUC delta: {auc_delta:+.4f}
- F1 delta: {f1_delta:+.4f}
- Degree-probability correlation: {deg_corr}

## Your Task
Write a concise JSON summary with:
- key_observation: 1 sentence synthesizing the most important finding
- recommendation: what MetaAgent should tell LLMFabAgent to focus on next round

Output ONLY the JSON.
"""
        try:
            result = await self.autonomous_thinking(
                prompt=prompt, state=None,
                system_message=SYSTEM_PROMPT, max_iterations=1,
            )
            parsed = self._extract_json(result.get("response", ""))
            if isinstance(parsed, dict):
                return {
                    "top_disagreements": disagreements[:5],
                    "uncertain_genes": uncertain[:10],
                    "pubmed_discordant": pubmed_discordant[:5],
                    "improvement": {"auc_delta": auc_delta, "f1_delta": f1_delta},
                    "key_observation": parsed.get("key_observation", ""),
                    "recommendation": parsed.get("recommendation", ""),
                }
        except Exception:
            pass

        return {
            "top_disagreements": disagreements[:5],
            "uncertain_genes": uncertain[:10],
            "improvement": {"auc_delta": auc_delta, "f1_delta": f1_delta},
            "key_observation": f"Iteration {iteration} complete.",
            "recommendation": "Continue refinement.",
        }

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
