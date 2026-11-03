"""MetaAgent — 首席科学家编排器：规划实验、复盘迭代并产出结论。"""

import json
import re
from typing import Any

from core.base_agent import BaseAgent
from core.config import Settings


# Dataset catalog (injected into LLM prompt)


DATASET_CATALOG = """
## Available Gene Interaction Network Datasets
| Dataset | Genes | Edges | Density | Type | Best For |
|---------|-------|-------|---------|------|----------|
| CPDB | 13,627 | 504,378 | 0.0027 | consensus | cancer, general disease genes |
| STRINGdb | 13,179 | 673,099 | 0.0039 | functional | pathway/functional analysis |
| PCNET | 19,781 | 5,449,448 | 0.0139 | large-scale | unbiased screening, rare diseases |
| LTG | 18,358 | 3,954,680 | 0.0117 | liver tissue | liver cancer, drug metabolism |
| MTG | ~26,000 | — | — | multi-tissue | tissue-specific comparisons |
| MULTINET | 14,398 | 219,134 | 0.0011 | high-confidence | strict filtering, strong evidence only |
| IREF | 17,013 | 743,137 | 0.0026 | experimental | experimentally validated interactions |
| IREF_2015 | 12,129 | 183,618 | 0.0012 | experimental | smaller validated network |
"""

# System Prompt


SYSTEM_PROMPT = """You are MetaAgent — Chief Scientist of the GeneAgent system.

## Role
You orchestrate gene-disease association prediction experiments. You:
1. Parse research goals, match to the best available dataset, and create execution plans
2. Review results after each iteration and decide next steps
3. Generate comprehensive final scientific conclusions

## Available Datasets
You have the datasets shown in the user prompt. Choose wisely based on the research goal.

## Mode Selection
After analyzing the goal, choose ONE of two modes:

### network_enhanced
Use when: the disease has a matching dataset (e.g., liver cancer → LTG, general cancer → CPDB).
Pipeline: GeneDataAgent reads network → LLMFabAgent predicts + GPS ensemble → PubMed validation.
Best for: common cancers, liver diseases, diseases with tissue-specific networks.

### llm_only
Use when: NO dataset matches the disease (e.g., Alzheimer's, Huntington's, COVID-19).
Pipeline: Skip network entirely → LLMFabAgent generates gene list from training knowledge → PubMed validation only.
Best for: rare diseases, neurological disorders, emerging diseases with no pre-built network.

## Output Format for Initial Plan
Return JSON:
{
  "mode": "network_enhanced" or "llm_only",
  "dataset": "CPDB" or "LTG" or ... or null (if llm_only),
  "disease_context": "<specific disease name>",
  "gene_categories": ["<category1>", "<category2>"],
  "reason": "<why this mode/dataset was chosen>"
}
"""


class MetaAgent(BaseAgent):
    """Chief Scientist orchestrator for gene-disease prediction."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(name="MetaAgent", settings=settings)

    def _get_system_prompt(self, state, default_prompt=None):
        return SYSTEM_PROMPT

    # ENTRY POINT 1: initial planning (first call)



    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Entry point: initial goal analysis (first call) OR iteration review."""
        iteration = state.get("iteration", 0)

        if iteration == 0:
            return await self._initial_plan(state)
        else:
            return await self._review_iteration(state)

    async def _initial_plan(self, state: dict[str, Any]) -> dict[str, Any]:
        """Parse goal, match dataset, decide mode."""
        goal = state.get("goal", "")
        forced_dataset = state.get("forced_dataset")
        print(f"\n [MetaAgent] Planning: {goal}")

        # Forced dataset: skip LLM, directly use specified dataset


        if forced_dataset:
            print(f"   🔒 Dataset forced to: {forced_dataset}")
            plan = {
                "mode": "network_enhanced",
                "dataset": forced_dataset,
                "disease_context": state.get("disease_context", "") or "cancer",
                "gene_categories": ["oncogenes", "tumor suppressor genes"],
                "reason": f"Dataset forced by user: {forced_dataset}",
            }
            print(f"   Mode: network_enhanced | Dataset: {forced_dataset}")
            return {
                "plan": plan,
                "disease_context": plan["disease_context"],
                "mode": "network_enhanced",
                "iteration": 1,
            }

        if not self.llm:
            # fallback: default to CPDB + network_enhanced


            return {
                "plan": {"mode": "network_enhanced", "dataset": "CPDB",
                         "disease_context": "cancer", "gene_categories": [],
                         "reason": "No LLM available, using defaults"},
                "disease_context": "cancer",
                "mode": "network_enhanced",
                "iteration": 1,
            }

        prompt = f"""# Research Goal
{goal}

{DATASET_CATALOG}

## Your Task
1. Extract the disease from the goal
2. Match it to the best dataset above
3. If a match exists → mode="network_enhanced", pick the dataset
4. If NO match → mode="llm_only", dataset=null
5. Identify relevant gene categories

## Matching Rules
- Liver cancer / liver disease / drug metabolism → LTG
- General cancer / breast cancer / lung cancer / etc. → CPDB (best cancer benchmark)
- Functional / pathway analysis → STRINGdb
- Unbiased / large-scale screening → PCNET
- Strict / high-confidence filtering → MULTINET
- Alzheimer's / Parkinson's / neurological → NO match → llm_only
- Rare genetic disease / emerging disease → NO match → llm_only

Return ONLY JSON:
{{"mode": "network_enhanced"|"llm_only", "dataset": "<name or null>", "disease_context": "<disease>", "gene_categories": ["..."], "reason": "<1 sentence>"}}"""

        try:
            result = await self.autonomous_thinking(
                prompt=prompt, state=None,
                system_message="You are a bioinformatics scientist. Output ONLY JSON.",
                max_iterations=1,
            )
            parsed = self._extract_json(result.get("response", ""))
            if isinstance(parsed, dict):
                mode = parsed.get("mode", "network_enhanced")
                dataset = parsed.get("dataset", "CPDB")
                disease_context = parsed.get("disease_context", "")
                gene_categories = parsed.get("gene_categories", [])
                reason = parsed.get("reason", "")

                plan = {
                    "mode": mode,
                    "dataset": dataset,
                    "disease_context": disease_context,
                    "gene_categories": gene_categories,
                    "reason": reason,
                }
                print(f"   Mode: {mode} | Dataset: {dataset} | Disease: {disease_context}")
                if gene_categories:
                    print(f"   Gene categories: {', '.join(gene_categories)}")
                print(f"   Reason: {reason}")

                return {
                    "plan": plan,
                    "disease_context": disease_context,
                    "mode": mode,
                    "iteration": 1,
                }
        except Exception as e:
            print(f"   ️ Plan parsing failed: {e}")

        # fallback


        return {
            "plan": {"mode": "network_enhanced", "dataset": "CPDB",
                     "disease_context": "cancer", "gene_categories": [],
                     "reason": "Fallback default"},
            "disease_context": "cancer",
            "mode": "network_enhanced",
            "iteration": 1,
        }

    # ENTRY POINT 2: review iteration results



    async def _review_iteration(self, state: dict[str, Any]) -> dict[str, Any]:
        """Review iteration results and decide CONTINUE or FINISH."""
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", 3)
        analysis_report = state.get("analysis_report", "")
        memory_insights = state.get("memory_insights", "")

        print(f"\n [MetaAgent] Reviewing iteration {iteration}/{max_iter}")

        # parse analysis for key metrics


        analysis_data = {}
        try:
            analysis_data = json.loads(analysis_report) if isinstance(analysis_report, str) else analysis_report
        except (json.JSONDecodeError, TypeError):
            pass

        ev = analysis_data.get("evaluation", {})
        proposed = ev.get("proposed_ensemble", {})
        current_auc = proposed.get("auc", 0) or 0

        # Parse memory_insights for structured gene lists


        mem_data = {}
        try:
            mem_data = json.loads(memory_insights) if isinstance(memory_insights, str) else (memory_insights or {})
        except Exception:
            pass
        uncertain_genes = mem_data.get("uncertain_genes", [])
        disagreements = mem_data.get("top_disagreements", [])
        pubmed_discordant = mem_data.get("pubmed_discordant", [])

        # check history for improvement


        iteration_history = state.get("iteration_history", [])
        prev_auc = 0
        if len(iteration_history) >= 2:
            prev_ev = iteration_history[-2].get("evaluation", {})
            prev_prop = prev_ev.get("proposed_ensemble", {})
            prev_auc = prev_prop.get("auc", 0) or 0

        if self.llm:
            decision, reason, llm_feedback = await self._llm_decide(
                iteration, max_iter, current_auc, prev_auc,
                analysis_data, mem_data, state.get("goal", ""),
            )
        else:
            if iteration >= max_iter:
                decision, reason = "FINISH", "Reached max iterations"
            elif current_auc > prev_auc + 0.01:
                decision, reason = "CONTINUE", f"AUC improved ({prev_auc:.4f} → {current_auc:.4f})"
            else:
                decision, reason = "FINISH", "AUC stable"
            llm_feedback = ""

        # Build structured feedback: LLM text + concrete gene lists


        feedback_parts = [llm_feedback] if llm_feedback else []
        if uncertain_genes:
            feedback_parts.append(f"UNCERTAIN_GENES: {json.dumps(uncertain_genes[:20])}")
        if disagreements:
            disagreement_names = [d.get("gene", d) for d in disagreements[:10]]
            feedback_parts.append(f"LLM_GPS_DISAGREEMENTS: {json.dumps(disagreement_names)}")
        if pubmed_discordant:
            discordant_names = [d.get("gene", d) for d in pubmed_discordant[:10]]
            feedback_parts.append(f"PUBMED_DISCORDANT: {json.dumps(discordant_names)}")
        feedback = "\n".join(feedback_parts)

        print(f"   Decision: {decision} — {reason}")

        # record this iteration in history


        iteration_record = {
            "iteration": iteration,
            "evaluation": ev,
            "decision": decision,
            "reason": reason,
        }
        iteration_history.append(iteration_record)

        return {
            "meta_decision": decision,
            "feedback": feedback,
            "iteration_history": iteration_history,
            "iteration": iteration + 1 if decision == "CONTINUE" else iteration,
        }

    async def _llm_decide(self, iteration, max_iter, current_auc, prev_auc,
                          analysis_data, mem_data, goal):
        """Use LLM to make the continue/finish decision."""
        align = analysis_data.get("alignment_summary", {})
        topo = analysis_data.get("topology_analysis", {})

        # structured data from MemoryAgent


        uncertain = mem_data.get("uncertain_genes", [])
        disagreements = mem_data.get("top_disagreements", [])
        pubmed_disc = mem_data.get("pubmed_discordant", [])
        key_obs = mem_data.get("key_observation", "")

        prompt = f"""# Iteration {iteration}/{max_iter} Review

## Metrics
- Current AUC: {current_auc:.4f}
- Previous AUC: {prev_auc:.4f}
- Alignment (LLM vs PubMed concordance): {align.get('concordance_rate', '?')}

## MemoryAgent Findings
- Key observation: {key_obs}
- Uncertain genes (prob ~0.5): {json.dumps(uncertain[:15])}
- LLM-GPS disagreements: {json.dumps(disagreements[:5])}
- PubMed discordant: {json.dumps(pubmed_disc[:5])}

## Decision
Should we CONTINUE (refine predictions with feedback) or FINISH?

Output:
DECISION: [CONTINUE] or [FINISH]
REASON: <1 sentence>
FEEDBACK: <if CONTINUE: specify what LLMFabAgent should focus on>
"""
        try:
            result = await self.autonomous_thinking(
                prompt=prompt, state=None,
                system_message=SYSTEM_PROMPT, max_iterations=1,
            )
            text = result.get("response", "")
            decision = "CONTINUE" if "CONTINUE" in text.upper() else "FINISH"
            reason = self._extract_line(text, "REASON:")
            feedback = self._extract_line(text, "FEEDBACK:")
            return decision, reason, feedback
        except Exception:
            if iteration >= max_iter:
                return "FINISH", "Max iterations reached (LLM failed)", ""
            return "CONTINUE", "Continue refinement (LLM failed, default)", ""

    # ENTRY POINT 3: final conclusion



    async def generate_final_conclusion(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive final research conclusion."""
        print(f"\n{'='*60}")
        print(f"🏁 [MetaAgent] Generating Final Conclusion...")
        print(f"{'='*60}")

        goal = state.get("goal", "")
        plan = state.get("plan", {})
        fab_results = state.get("fab_results", {})
        literature_report = state.get("literature_report", "")
        analysis_report = state.get("analysis_report", "")
        iteration_history = state.get("iteration_history", [])

        # parse structured data


        analysis_data = {}
        try:
            analysis_data = json.loads(analysis_report) if isinstance(analysis_report, str) else analysis_report
        except Exception:
            pass

        if not self.llm:
            return {"final_conclusion": self._fallback_conclusion(state)}

        # extract top predictions


        predictions = fab_results.get("predictions", {}) if isinstance(fab_results, dict) else {}
        top_genes = sorted(
            [(s, p) for s, p in predictions.items() if isinstance(p, dict)],
            key=lambda x: x[1].get("probability", 0) or 0, reverse=True,
        )[:15]

        top_table = "\n".join(
            f"| {sym:<10s} | {(pred.get('probability') or 0):.4f} | "
            f"{pred.get('confidence', '?'):<6s} | {pred.get('degree_tier', '?'):<10s} |"
            for sym, pred in top_genes
        )

        ev = analysis_data.get("evaluation", {})
        baseline = ev.get("baseline_llm_only", {})
        proposed = ev.get("proposed_ensemble", {})
        align = analysis_data.get("alignment_summary", {})

        # Cross-round summary


        all_round_results = state.get("all_round_results", [])
        round_lines = []
        for r in all_round_results:
            round_lines.append(
                f"| R{r.get('iteration', '?')} | {r.get('round_label', '?')} | "
                f"AUC={r.get('evaluation_auc', '?')}, F1={r.get('evaluation_f1', '?')} | "
                f"{r.get('n_disagreements', '?')} disagreements, {r.get('n_uncertain', '?')} uncertain |"
            )
        round_table = "\n".join(round_lines) if round_lines else "Single round only"

        prompt = f"""# Final Research Conclusion

## Goal
{goal}

## Dataset
{json.dumps(plan.get('disease_context', '') or 'CPDB gene interaction network')}

## Iterations Completed
{len(iteration_history)}

## Round-by-Round Evolution
| Round | Strategy | Key Metrics | Issues |
|-------|----------|-------------|--------|
{round_table}

## Final Metrics
| Method | AUC | F1 |
|--------|-----|-----|
| Baseline (LLM only) | {baseline.get('auc', 'N/A')} | {baseline.get('f1', 'N/A')} |
| Proposed (LLM+GPS) | {proposed.get('auc', 'N/A')} | {proposed.get('f1', 'N/A')} |

## PubMed Cross-Validation
- Concordance rate: {align.get('concordance_rate', 0)}
- Concordant genes: {align.get('concordant', '?')} / {align.get('total_genes', '?')}

## Top Predicted Disease-Associated Genes
| Gene | Probability | Confidence | Network Tier |
|------|-------------|------------|--------------|
{top_table}

## Literature Evidence
{str(literature_report)[:2000]}

## Your Task
Write a comprehensive final research conclusion as a Markdown report covering:
1. **Research Summary**: goal, dataset, and approach
2. **Key Findings**: top genes and their biological significance
3. **LLM vs GPS**: how did the ensemble improve predictions?
4. **PubMed Validation**: how well did LLM knowledge match real literature?
5. **Network Insights**: did hub genes predict better?
6. **Recommendations**: what should researchers do next?
7. **Limitations**: what are the caveats?

Write in professional scientific language. Cite specific genes and numbers.
"""
        try:
            result = await self.autonomous_thinking(
                prompt=prompt, state=None,
                system_message=SYSTEM_PROMPT, max_iterations=1,
            )
            conclusion = result.get("response", "")
        except Exception as e:
            conclusion = f"Conclusion generation failed: {e}"

        return {"final_conclusion": conclusion}

    def _fallback_conclusion(self, state: dict[str, Any]) -> str:
        """Generate a basic conclusion without LLM."""
        fab = state.get("fab_results", {})
        predictions = fab.get("predictions", {}) if isinstance(fab, dict) else {}
        top = sorted(
            [(s, p.get("probability", 0)) for s, p in predictions.items() if isinstance(p, dict)],
            key=lambda x: x[1], reverse=True,
        )[:10]
        lines = [
            "# GeneAgent — Gene-Disease Association Prediction Results",
            f"Goal: {state.get('goal', '')}",
            f"Top predicted disease-associated genes:",
        ]
        for sym, prob in top:
            lines.append(f"- {sym}: {prob:.4f}")
        return "\n".join(lines)

    # Helpers



    def _extract_json(self, text: str) -> Any:
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

    def _extract_line(self, text: str, prefix: str) -> str:
        for line in text.split("\n"):
            if line.strip().upper().startswith(prefix.upper()):
                return line.strip()[len(prefix):].strip()
        return ""
