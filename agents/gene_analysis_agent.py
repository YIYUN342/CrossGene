"""GeneAnalysisAgent — 生物学解释与定量评估（真实数据库交叉验证 + 拓扑分析）。"""

import json
from typing import Any

from core.base_agent import BaseAgent
from core.config import Settings

# Reuse pubmed_evidence from LiteratureAgent (no direct API calls)



SYSTEM_PROMPT = """You are GeneAnalysisAgent — Biological Interpretation Specialist.

## Core Mission
Interpret gene-disease prediction results using data from:
1. LiteratureAgent's PubMed publication counts (already collected)
2. Network topology analysis (degree, tier, hub status)
3. Your training knowledge of disease pathways and mechanisms

## Analysis Framework
- **Known vs Novel**: Is this gene supported by real PubMed evidence?
- **Network Role**: Hub vs peripheral — does topology matter?
- **Biological Plausibility**: Based on real data, not speculation
- **Confidence Assessment**: How much should we trust each prediction?

## Output Rules
- Base ALL claims on tool-returned data
- If a tool returns no data for a gene, say "no data" — don't invent
- Compare prediction probability vs real PubMed evidence
- Identify cases where LLM prediction conflicts with real data
"""


class GeneAnalysisAgent(BaseAgent):
    """Biological interpretation agent — metrics + topology only (no external APIs)."""

    def __init__(self, settings: Settings | None = None):
        super().__init__(name="GeneAnalysisAgent", settings=settings)

    def _get_system_prompt(self, state, default_prompt=None):
        return SYSTEM_PROMPT

    # Main run



    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        print(f"\n{'='*60}")
        print(f" [GeneAnalysisAgent] Quantitative Evaluation + Real-DB Cross-Reference")
        print(f"{'='*60}")

        goal = state.get("goal", "")
        mode = state.get("mode", "network_enhanced")
        fab_results = state.get("fab_results", {})
        data_context = state.get("data_context", "")

        # 1. Extract predictions (baseline + proposed)


        baseline_preds = {}
        proposed_preds = {}
        if isinstance(fab_results, dict):
            baseline_preds = fab_results.get("baseline_llm_only", {})
            proposed_preds = fab_results.get("proposed_ensemble", {})
            if not proposed_preds:
                proposed_preds = fab_results.get("predictions", {})

        # 2. Compute metrics (skip in llm_only — no ground truth)


        baseline_metrics = {}
        proposed_metrics = {}
        if mode != "llm_only":
            baseline_metrics = self._compute_metrics(baseline_preds) if baseline_preds else {}
            proposed_metrics = self._compute_metrics(proposed_preds) if proposed_preds else {}

            # Print comparison


            print(f"\n   {'='*55}")
            print(f"   {'Method':<25s} {'AUC':>8s} {'AUPR':>8s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s}")
            print(f"   {'-'*55}")
            for name, m in [("Baseline (LLM only)", baseline_metrics),
                             ("Proposed (LLM+GPS)", proposed_metrics)]:
                if m:
                    print(f"   {name:<25s} {str(m.get('auc') or 'N/A'):>8s} "
                          f"{str(m.get('aupr') or 'N/A'):>8s} "
                          f"{str(m.get('precision') or 'N/A'):>8s} "
                          f"{str(m.get('recall') or 'N/A'):>8s} "
                          f"{str(m.get('f1') or 'N/A'):>8s}")

            if baseline_metrics and proposed_metrics:
                delta_auc = (proposed_metrics.get('auc') or 0) - (baseline_metrics.get('auc') or 0)
                delta_f1 = (proposed_metrics.get('f1') or 0) - (baseline_metrics.get('f1') or 0)
                print(f"\n   Improvement: ΔAUC={delta_auc:+.4f}, ΔF1={delta_f1:+.4f}")
        else:
            print(f"\n   ═══ LLM-Only Mode: no ground truth → skip AUC computation ═══")

        # 3. Per-gene ranking & topology analysis


        primary_preds = proposed_preds if proposed_preds else baseline_preds

        ranked_genes = sorted(
            [(sym, pred) for sym, pred in primary_preds.items()
             if isinstance(pred, dict) and pred.get("probability") is not None],
            key=lambda x: x[1].get("probability", 0),
            reverse=True,
        )

        # Topology analysis (skip in llm_only — no network data)


        topology_stats = {}
        if mode != "llm_only":
            topology_stats = self._analyze_topology_effect(ranked_genes)

        # 4. Read PubMed evidence (prefer structured; fallback to regex)


        pubmed_evidence = state.get("pubmed_evidence") or {}
        if not pubmed_evidence:
            # fallback: parse from LiteratureAgent's markdown report


            literature_report = state.get("literature_report", "")
            import re as _re
            per_gene_match = _re.search(r'PUBMED_PER_GENE (\{.*?\})', str(literature_report))
            if per_gene_match:
                try:
                    pubmed_evidence = json.loads(per_gene_match.group(1))
                except (json.JSONDecodeError, AttributeError):
                    pass

        n_queried = len(pubmed_evidence)
        n_high = sum(1 for c in pubmed_evidence.values() if c > 100)
        print(f"\n    LiteratureAgent provided PubMed data for {n_queried} genes "
              f"({n_high} high-evidence)")

        # 5. Per-gene structured analysis


        analyzed_genes = ranked_genes[:20]
        per_gene_analysis = []
        for sym, pred in analyzed_genes:
            if not isinstance(pred, dict):
                continue
            entry = {
                "gene_name": sym,
                "gene_id": pred.get("gene_id"),
                "probability": pred.get("probability"),
                "confidence": pred.get("confidence"),
                "evidence_type": pred.get("evidence_type"),
                "degree_tier": pred.get("degree_tier"),
                "gene_degree": pred.get("gene_degree"),
                "ground_truth": pred.get("ground_truth"),
                "source": pred.get("source"),
                "gps_score": pred.get("gps_score"),
                "pubmed_count": pubmed_evidence.get(sym),
            }

            # Classify: LLM prediction vs real evidence alignment


            llm_prob = entry.get("probability") or 0
            pubmed_count = entry.get("pubmed_count")

            if pubmed_count is None:
                entry["alignment"] = "no_api_data"
            elif pubmed_count >= 100 and llm_prob >= 0.7:
                entry["alignment"] = "concordant_high"      # both agree: strong signal
            elif pubmed_count < 10 and llm_prob >= 0.7:
                entry["alignment"] = "llm_high_db_low"     # LLM overconfident, no real evidence
            elif pubmed_count >= 100 and llm_prob <= 0.3:
                entry["alignment"] = "llm_low_db_high"     # LLM missed a well-studied gene
            elif pubmed_count < 10 and llm_prob <= 0.3:
                entry["alignment"] = "concordant_low"       # both agree: weak signal
            else:
                entry["alignment"] = "concordant_medium"

            per_gene_analysis.append(entry)

        # 6. LLM interpretation


        llm_interpretation = ""
        if self.llm:
            # check previous round for comparison


            prev_metrics = None
            iteration_history = state.get("iteration_history", [])
            if iteration_history:
                prev_entry = iteration_history[-1]
                prev_ev = prev_entry.get("evaluation", {})
                prev_metrics = prev_ev.get("proposed_ensemble", {})

            llm_interpretation = await self._llm_interpret(
                goal=goal,
                baseline_metrics=baseline_metrics,
                proposed_metrics=proposed_metrics,
                prev_metrics=prev_metrics,
                topology_stats=topology_stats,
                pubmed_evidence=pubmed_evidence,
                per_gene=per_gene_analysis[:20],
            )

        # 7. Cross-round comparison


        all_round_results = state.get("all_round_results", [])
        cross_round = self._compare_rounds(all_round_results)

        # 8. Build final structured output


        analysis_output = {
            "evaluation": {
                "baseline_llm_only": baseline_metrics,
                "proposed_ensemble": proposed_metrics,
            },
            "topology_analysis": topology_stats,
            "database_cross_reference": {
                "source": "LiteratureAgent (PubMed + MyGene + Open Targets API)",
                "genes_with_pubmed_data": len(pubmed_evidence),
                "high_evidence_genes": n_high,
            },
            "per_gene_analysis": per_gene_analysis,
            "alignment_summary": self._summarize_alignment(per_gene_analysis),
            "cross_round_comparison": cross_round,
            "llm_interpretation": llm_interpretation,
        }

        print(f"\n{'─'*60}")
        print(f" [GeneAnalysisAgent] Analysis Complete")
        print(f"   ├─ Baseline vs Proposed evaluation complete")
        print(f"   ├─ Topology: {topology_stats.get('n_genes_analyzed', 0)} genes")
        print(f"   ├─ PubMed data (from LiteratureAgent): {len(pubmed_evidence)} genes")
        print(f"   └─ Alignment: {analysis_output['alignment_summary']}")

        return {"analysis_report": json.dumps(analysis_output, indent=2, ensure_ascii=False)}

    # Quantitative metrics (deterministic, sklearn if available)



    def _compute_metrics(self, predictions: dict) -> dict:
        """Compute AUC, AUPR, precision, recall, F1 against ground truth."""
        import numpy as np

        scores, labels = [], []
        for sym, pred in predictions.items():
            if not isinstance(pred, dict):
                continue
            gt = pred.get("ground_truth")
            prob = pred.get("probability")
            if gt is not None and prob is not None:
                scores.append(float(prob))
                labels.append(int(gt))

        n = len(labels)
        if n < 2:
            return {"n_labeled": n, "note": "Insufficient labeled data"}

        scores_arr = np.array(scores)
        labels_arr = np.array(labels)
        pos_mask = labels_arr == 1
        neg_mask = labels_arr == 0
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())

        # AUC


        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            auc = round(float(roc_auc_score(labels_arr, scores_arr)), 4)
            aupr = round(float(average_precision_score(labels_arr, scores_arr)), 4)
        except ImportError:
            pos_scores = scores_arr[pos_mask]
            neg_scores = scores_arr[neg_mask]
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                n_correct = int(np.sum(pos_scores[:, None] > neg_scores[None, :]))
                n_ties = int(np.sum(pos_scores[:, None] == neg_scores[None, :]))
                auc = round((n_correct + 0.5 * n_ties) / (len(pos_scores) * len(neg_scores)), 4)
            else:
                auc = None
            aupr = None

        # Precision, Recall, F1 @ 0.5


        pred_pos = (scores_arr >= 0.5).astype(int)
        tp = int(np.sum((pred_pos == 1) & (labels_arr == 1)))
        fp = int(np.sum((pred_pos == 1) & (labels_arr == 0)))
        fn = int(np.sum((pred_pos == 0) & (labels_arr == 1)))
        tn = int(np.sum((pred_pos == 0) & (labels_arr == 0)))

        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
        accuracy = round((tp + tn) / n, 4)

        # Distribution stats


        pos_scores_arr = scores_arr[pos_mask]
        neg_scores_arr = scores_arr[neg_mask]

        return {
            "n_labeled": n,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "auc": auc,
            "aupr": aupr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "mean_prob_positive": round(float(pos_scores_arr.mean()), 4) if len(pos_scores_arr) > 0 else None,
            "mean_prob_negative": round(float(neg_scores_arr.mean()), 4) if len(neg_scores_arr) > 0 else None,
            "std_prob_positive": round(float(pos_scores_arr.std()), 4) if len(pos_scores_arr) > 0 else None,
            "std_prob_negative": round(float(neg_scores_arr.std()), 4) if len(neg_scores_arr) > 0 else None,
        }

    # Network topology analysis (deterministic)



    def _analyze_topology_effect(self, ranked_genes: list) -> dict:
        """Analyze how network topology (degree, tier) affects prediction quality."""
        by_tier = {}
        for sym, pred in ranked_genes:
            if not isinstance(pred, dict):
                continue
            tier = pred.get("degree_tier", "unknown")
            prob = pred.get("probability")
            gt = pred.get("ground_truth")
            degree = pred.get("gene_degree", 0)
            gps_score = pred.get("gps_score")

            if tier not in by_tier:
                by_tier[tier] = {"count": 0, "probs": [], "degrees": [],
                                 "ground_truths": [], "gps_scores": []}
            by_tier[tier]["count"] += 1
            if prob is not None:
                by_tier[tier]["probs"].append(prob)
            by_tier[tier]["degrees"].append(degree)
            if gt is not None:
                by_tier[tier]["ground_truths"].append(gt)
            if gps_score is not None:
                by_tier[tier]["gps_scores"].append(gps_score)

        tier_summary = {}
        for tier, data in sorted(by_tier.items()):
            probs = data["probs"]
            gts = data["ground_truths"]
            mean_prob = round(sum(probs) / len(probs), 4) if probs else 0
            mean_degree = round(sum(data["degrees"]) / len(data["degrees"]), 1) if data["degrees"] else 0
            precision = round(sum(1 for p, g in zip(probs, gts)
                                  if p >= 0.5 and g == 1) / max(1, sum(1 for p in probs if p >= 0.5)), 4) if probs else 0
            mean_gps = round(sum(data["gps_scores"]) / len(data["gps_scores"]), 4) if data["gps_scores"] else None

            tier_summary[tier] = {
                "n_genes": data["count"],
                "mean_probability": mean_prob,
                "mean_degree": mean_degree,
                "precision_at_0.5": precision,
                "mean_gps_score": mean_gps,
            }

        # Correlation: degree vs probability


        degrees = []
        probs = []
        for sym, pred in ranked_genes:
            if isinstance(pred, dict):
                d = pred.get("gene_degree")
                p = pred.get("probability")
                if d is not None and p is not None:
                    degrees.append(d)
                    probs.append(p)

        degree_prob_corr = None
        if len(degrees) > 2:
            import numpy as np
            try:
                degree_prob_corr = round(float(np.corrcoef(degrees, probs)[0, 1]), 4)
            except Exception:
                pass

        return {
            "n_genes_analyzed": len(ranked_genes),
            "by_degree_tier": tier_summary,
            "degree_probability_correlation": degree_prob_corr,
        }

    # Cross-round comparison



    def _compare_rounds(self, all_round_results: list) -> dict:
        """Compare AUC/F1 trends across rounds."""
        if not all_round_results or len(all_round_results) < 2:
            return {"rounds": len(all_round_results), "note": "Need ≥2 rounds for comparison"}

        rounds_data = []
        prev_auc = None
        for r in all_round_results:
            auc = r.get("evaluation_auc")
            f1 = r.get("evaluation_f1")
            auc_delta = round(auc - prev_auc, 4) if prev_auc is not None and auc is not None else None
            rounds_data.append({
                "iteration": r.get("iteration"),
                "auc": auc,
                "f1": f1,
                "auc_delta": auc_delta,
                "n_disagreements": r.get("n_disagreements"),
                "n_uncertain": r.get("n_uncertain"),
                "n_pubmed_discordant": r.get("n_pubmed_discordant"),
            })
            prev_auc = auc

        # overall trend


        first_auc = rounds_data[0].get("auc", 0) or 0
        last_auc = rounds_data[-1].get("auc", 0) or 0
        overall_delta = round(last_auc - first_auc, 4)

        return {
            "rounds": len(all_round_results),
            "overall_auc_delta": overall_delta,
            "trend": "improving" if overall_delta > 0.005 else "stable" if abs(overall_delta) <= 0.005 else "declining",
            "per_round": rounds_data,
        }

    # LLM interpretation (anchored to real data)



    async def _llm_interpret(
        self,
        goal: str,
        baseline_metrics: dict,
        proposed_metrics: dict,
        prev_metrics: dict = None,
        topology_stats: dict = None,
        pubmed_evidence: dict = None,
        per_gene: list = None,
    ) -> str:
        """Generate LLM interpretation based on REAL data."""

        pubmed_evidence = pubmed_evidence or {}
        topology_stats = topology_stats or {}
        per_gene = per_gene or []
        prev_metrics = prev_metrics or {}

        eval_data = {
            "baseline_llm_only": baseline_metrics,
            "proposed_ensemble_llm_gps": proposed_metrics,
        }
        if prev_metrics:
            eval_data["previous_round_best"] = {
                "auc": prev_metrics.get("auc"),
                "f1": prev_metrics.get("f1"),
            }

        prompt = f"""# Gene Prediction Analysis — Data-Driven Interpretation

**Research Goal**: {goal}

## Quantitative Metrics (computed from predictions vs ground truth)
```json
{json.dumps(eval_data, indent=2)}
```

## Network Topology Impact (deterministic analysis)
```json
{json.dumps(topology_stats, indent=2)}
```

## Real Database Evidence (PubMed API — from LiteratureAgent)
```json
{json.dumps({"genes_with_pubmed_data": len(pubmed_evidence), "high_evidence": sum(1 for c in pubmed_evidence.values() if c > 100), "sample": dict(list(pubmed_evidence.items())[:5])}, indent=2)}
```

## Top Gene Analysis (prediction + real DB evidence merged)
```json
{json.dumps(per_gene[:15], indent=2)}
```

## Your Task
Based ONLY on the data shown above, provide a concise interpretation:

1. **Prediction Quality** (1-2 sentences): Which round performs best? Is the model well-calibrated?
2. **Round-over-Round Improvement** (1-2 sentences): Did LLM+GPS ensemble improve over baseline? If previous round data exists, did we improve?
3. **Network Topology Effect** (1-2 sentences): Does degree tier affect prediction accuracy?
4. **LLM vs Real Data** (2-3 sentences): Where do LLM predictions agree/disagree with real PubMed evidence?
5. **Key Findings** (3-5 bullets): Most important insights from this analysis
6. **Recommendations** (1-2 sentences): What should be prioritized for follow-up?

Output in Markdown. Be specific — cite the actual numbers from the data above.
DO NOT make claims not supported by the data.
"""
        try:
            # Use simple invoke (no tools) — real data already collected above


            if self.llm:
                response = await self.llm.ainvoke_simple(prompt, system_message=SYSTEM_PROMPT)
                return response if response else ""
            return ""
        except Exception as e:
            return f"LLM interpretation failed: {e}"

    # Alignment summary



    def _summarize_alignment(self, per_gene: list) -> dict:
        """Summarize LLM-vs-real-DB alignment across all analyzed genes."""
        counts = {}
        for entry in per_gene:
            align = entry.get("alignment", "unknown")
            counts[align] = counts.get(align, 0) + 1

        total = len(per_gene)
        concordant = counts.get("concordant_high", 0) + counts.get("concordant_low", 0) + counts.get("concordant_medium", 0)
        discordant = counts.get("llm_high_db_low", 0) + counts.get("llm_low_db_high", 0)
        no_data = counts.get("no_api_data", 0)

        # concordance rate is only meaningful for genes WITH PubMed data


        with_data = total - no_data
        return {
            "total_genes": total,
            "genes_with_pubmed_data": with_data,
            "concordant": concordant,
            "discordant": discordant,
            "concordance_rate": round(concordant / with_data, 4) if with_data > 0 else 0,
            "breakdown": counts,
        }

    # 不覆盖 autonomous_thinking：无需工具，外部 API 由 LiteratureAgent 处理。


