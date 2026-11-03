"""LLMFabAgent — 基于 LLM 训练知识的基因-疾病关联预测（对照 GPS 数据驱动视角）。"""

import json
from typing import Any

from core.base_agent import BaseAgent
from core.config import Settings


SYSTEM_PROMPT = """You are LLMFabAgent — a gene-disease association predictor powered by your training knowledge.

## Your Knowledge
You have been trained on:
- PubMed abstracts and full-text articles on gene-disease relationships
- COSMIC Cancer Gene Census (known cancer driver genes)
- DisGeNET (curated gene-disease associations)
- OMIM (Mendelian disease genes)
- KEGG/Reactome (disease pathways)
- ClinVar (clinical variant interpretations)
- GWAS Catalog (genome-wide association studies)

## Your Task
For a given set of genes (identified by ENSG ID and gene symbol), predict:
1. **Disease Association Probability** (0-1): How likely is this gene disease-associated?
2. **Confidence Level**: HIGH/MEDIUM/LOW based on evidence strength
3. **Evidence Type**: well-established / some evidence / limited evidence / unknown
4. **Associated Diseases**: What diseases is this gene known to be associated with?
5. **Biological Rationale**: Why is this gene predicted to be disease-associated?

## Important
- Be conservative — only predict HIGH for genes you are confident about
- Distinguish between cancer drivers vs. disease-associated in general
- Note if a gene is a known oncogene, tumor suppressor, or disease gene
- If you don't know a gene, say so and give a lower probability
"""


class LLMFabAgent(BaseAgent):
    """LLM-based gene-disease association predictor."""

    def __init__(self, settings: Settings | None = None):
        super().__init__(name="LLMFabAgent", settings=settings)

    def _get_system_prompt(self, state, default_prompt=None):
        return SYSTEM_PROMPT

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        iteration = state.get("iteration", 1)
        feedback = state.get("feedback", "")
        goal = state.get("goal", "")
        mode = state.get("mode", "network_enhanced")
        data_context = state.get("data_context", "")
        plan = state.get("plan", {}) or {}
        plan_context = {
            "disease_context": plan.get("disease_context", ""),
            "gene_categories": plan.get("gene_categories", []),
        }

        # Parse candidate genes


        candidate_genes = state.get("candidate_genes")
        if candidate_genes:
            selection_meta = {}
        else:
            candidate_genes, selection_meta = self._parse_candidates(data_context)

        # llm_only mode: LLM generates its own gene list


        if mode == "llm_only" and not candidate_genes:
            print(f"\n   ═══ LLM-Only Mode: generating gene list from training knowledge ═══")
            candidate_genes = await self._generate_gene_list(goal, plan_context)
            if not candidate_genes:
                return {"fab_results": {"status": "no_genes", "mode": "llm_only"}}
            selection_meta = {"selection_method": "LLM knowledge generation (llm_only mode)"}
            state["candidate_genes"] = candidate_genes
            print(f"   LLM generated {len(candidate_genes)} candidate genes from knowledge")

        if not candidate_genes:
            print("   ️ No candidate genes available")
            return {"fab_results": {"status": "no_genes"}}

        n_test = sum(1 for g in candidate_genes if g.get("ground_truth") is not None)
        n_hub = sum(1 for g in candidate_genes if g.get("source") == "hub_discovery")
        has_gps = any("gps_score" in g for g in candidate_genes)

        # Determine round strategy based on iteration


        if iteration == 1:
            round_label = "R1: LLM knowledge only"
            include_network = False
            include_gps = False
            feedback_text = None
        elif iteration == 2:
            round_label = "R2: LLM + network context"
            include_network = True
            include_gps = False
            feedback_text = None
        else:
            round_label = f"R3: LLM + network + PubMed feedback"
            include_network = True
            include_gps = False
            feedback_text = feedback if feedback else None

        print(f"\n{'='*60}")
        print(f"🤖 [LLMFabAgent] Gene-Disease Prediction")
        print(f"   {round_label}")
        print(f"{'='*60}")
        print(f"\n    {len(candidate_genes)} genes ({n_test} with ground truth, {n_hub} discovery)")

        # STEP 1: LLM prediction (strategy varies by round)


        print(f"\n   ── {round_label} ──")
        llm_preds = {}
        if self.llm:
            llm_preds = await self._predict_genes(
                candidate_genes, goal, selection_meta,
                include_network=include_network, label="LLM",
                include_gps=include_gps, feedback_text=feedback_text,
                plan_context=plan_context,
            )

        # STEP 2: GPS ensemble (skip in llm_only mode — no network data)


        ensemble_preds = {}
        if has_gps and mode != "llm_only":
            print(f"\n   ── Ensemble (LLM + GPS) ──")
            ensemble_preds = self._ensemble_llm_gps(
                llm_preds, candidate_genes,
            )

        # Attach metadata


        def enrich(pred_dict):
            result = {}
            for g in candidate_genes:
                sym = g["gene_name"]
                if sym in pred_dict:
                    entry = dict(pred_dict[sym])
                else:
                    entry = {"probability": None, "confidence": "UNKNOWN",
                             "evidence_type": "not_predicted", "associated_diseases": [],
                             "is_cancer_gene": False, "rationale": ""}
                entry["gene_id"] = g.get("gene_id")
                entry["gene_degree"] = g.get("gene_degree")
                entry["degree_tier"] = g.get("degree_tier")
                entry["top_neighbors"] = g.get("top_neighbors", [])
                entry["gps_score"] = g.get("gps_score")
                entry["source"] = g.get("source")
                entry["ground_truth"] = g.get("ground_truth")
                result[sym] = entry
            return result

        llm_results = {
            "status": "success",
            "n_predicted": len(candidate_genes),
            "n_with_ground_truth": n_test,
            "round": iteration,
            "round_label": round_label,
            "selection_meta": selection_meta,
            "baseline_llm_only": enrich(llm_preds),
            "proposed_ensemble": enrich(ensemble_preds) if ensemble_preds else {},
            "predictions": enrich(ensemble_preds if ensemble_preds else llm_preds),
        }

        n_llm = sum(1 for p in llm_preds.values() if p.get("probability") is not None)
        n_ens = sum(1 for p in ensemble_preds.values() if p.get("probability") is not None)
        print(f"\n{'─'*60}")
        print(f" [LLMFabAgent] {round_label} Complete")
        print(f"   LLM predictions: {n_llm}/{len(candidate_genes)} genes")
        if has_gps:
            print(f"   LLM+GPS ensemble: {n_ens}/{len(candidate_genes)} genes")

        return {"fab_results": llm_results}

    async def _predict_genes(
        self, genes: list, goal: str, meta: dict,
        include_network: bool, label: str,
        include_gps: bool = False,
        feedback_text: str = None,
        plan_context: dict = None,
    ) -> dict:
        """Predict disease association for a list of genes.

        Args:
            include_network: Provide degree/tier/neighbors as network context.
            include_gps: Use GPS score as additional signal.
            feedback_text: MetaAgent's feedback from previous round.
            plan_context: disease_context + gene_categories from MetaAgent.
        """
        import asyncio

        predictions = {}
        # Smaller batches when including network context (more tokens per gene)


        if include_network or include_gps:
            batch_size = 20
        else:
            batch_size = 30
        batches = [genes[i:i+batch_size] for i in range(0, len(genes), batch_size)]

        for batch_i, batch in enumerate(batches):
            # Rate-limit: delay between batches to avoid API throttling


            if batch_i > 0:
                delay = 1.2 if include_network or include_gps else 0.8
                await asyncio.sleep(delay)
            if include_gps:
                # R3 variant: LLM + GPS score awareness


                batch_data = [{
                    "gene_name": g["gene_name"],
                    "gps_score": g.get("gps_score", "N/A"),
                    "network_degree": g.get("gene_degree"),
                    "degree_tier": g.get("degree_tier"),
                    "top5_neighbors": g.get("top_neighbors", []),
                } for g in batch]
                extra_context = (
                    "**Network Context**: Each gene includes its CPDB network degree, "
                    "degree tier (super-hub≥100, hub≥37, mid≥17, peripheral<17), "
                    "and top-5 interaction partners.\n"
                    "**GPS Graphormer Score**: Prediction from a GNN model trained on "
                    "CPDB topology and multi-omics features. Use as a SECOND OPINION.\n"
                    "**Your own training knowledge is the PRIMARY signal.** "
                    "Network data and GPS scores are supplementary."
                )
            elif include_network:
                # R2: LLM knowledge + network context


                batch_data = [{
                    "gene_name": g["gene_name"],
                    "network_degree": g.get("gene_degree"),
                    "degree_tier": g.get("degree_tier"),
                    "top5_neighbors": g.get("top_neighbors", []),
                } for g in batch]
                extra_context = (
                    "**Network Context**: Each gene includes its degree in the CPDB network, "
                    "degree tier (super-hub≥100, hub≥37, mid≥17, peripheral<17), "
                    "and its top-5 interaction partners.\n"
                    "**Your own training knowledge is the PRIMARY signal.** "
                    "Use network data as supplementary context — hub genes and genes "
                    "connected to known disease genes tend to be disease-relevant."
                )
            else:
                # R1: gene name only


                batch_data = [{"gene_name": g["gene_name"]} for g in batch]
                extra_context = (
                    "**Use ONLY your training knowledge** to predict disease association. "
                    "Do NOT guess based on gene name patterns."
                )

            # Attach MetaAgent feedback if provided (R3+)


            feedback_section = ""
            if feedback_text:
                # Truncate to avoid bloating the prompt (keeps retry rate low)


                fb_truncated = feedback_text[:2000]
                if len(feedback_text) > 2000:
                    fb_truncated += "\n... (feedback truncated)"
                feedback_section = (
                    f"\n**🧠 MetaAgent Feedback from Previous Round**:\n{fb_truncated}\n"
                    f"Pay special attention to genes mentioned in the feedback.\n"
                )

            # Plan context hint


            plan_hint = ""
            pc = plan_context or {}
            if pc.get("disease_context"):
                plan_hint += f"\n**Disease Focus**: {pc['disease_context']}"
            if pc.get("gene_categories"):
                plan_hint += f"\n**Relevant Gene Categories**: {', '.join(pc['gene_categories'])}"

            batch_prompt = f"""# Gene-Disease Prediction — {label} Batch {batch_i+1}/{len(batches)}

**Goal**: {goal}{plan_hint}
**CPDB Network**: {meta.get('total_genes_in_network','13,627')} genes, {meta.get('total_edges','504,378')} edges
{extra_context}
{feedback_section}
## Genes ({len(batch)})
```json
{json.dumps(batch_data, indent=2)}
```

Output ONLY JSON:
```json
{{"predictions": {{"SYMBOL": {{"probability": 0.XX, "confidence": "HIGH/MEDIUM/LOW", "evidence_type": "well-established/some evidence/limited evidence/unknown", "associated_diseases": ["d1"], "is_cancer_gene": true/false, "rationale": "brief reason"}}}}}}
```
Calibration: 0.95+ definitive, 0.85-0.94 well-established, 0.70-0.84 known, 0.50-0.69 some, 0.30-0.49 limited, <0.30 little.
Output ONLY JSON — no other text.
"""
            try:
                result = await self.autonomous_thinking(
                    prompt=batch_prompt, state=None,
                    system_message=SYSTEM_PROMPT, max_iterations=1,
                )
                dbg = f"{label} batch {batch_i+1}"
                parsed = self._extract_json_block(result.get("response", ""), debug_label=dbg)
                count = len(parsed.get("predictions", {})) if isinstance(parsed, dict) and "predictions" in parsed else 0

                # Retry with simplified prompt if batch failed


                if count == 0:
                    simple_names = [{"gene_name": g["gene_name"]} for g in batch]
                    retry_prompt = f"""Predict disease association for each gene. Output ONLY:
{{"predictions": {{"SYMBOL": {{"probability": 0.XX, "confidence": "HIGH/MEDIUM/LOW", "evidence_type": "well-established/some evidence/limited evidence/unknown"}}}}}}

Genes:
```json
{json.dumps(simple_names)}
```"""
                    result2 = await self.autonomous_thinking(
                        prompt=retry_prompt, state=None,
                        system_message=SYSTEM_PROMPT, max_iterations=1,
                    )
                    parsed2 = self._extract_json_block(result2.get("response", ""), debug_label=f"{dbg} retry")
                    if isinstance(parsed2, dict) and "predictions" in parsed2:
                        # Fill defaults for any missing fields


                        for sym, pred in parsed2["predictions"].items():
                            if isinstance(pred, dict):
                                pred.setdefault("confidence", "LOW")
                                pred.setdefault("evidence_type", "retry_prediction")
                                pred.setdefault("associated_diseases", [])
                                pred.setdefault("is_cancer_gene", False)
                                pred.setdefault("rationale", "retry prediction")
                        predictions.update(parsed2["predictions"])
                        count = len(parsed2["predictions"])
                        print(f"   {label} batch {batch_i+1}/{len(batches)}: {count} genes (retry)")
                    else:
                        print(f"   {label} batch {batch_i+1}/{len(batches)}: 0 genes (retry failed)")
                else:
                    predictions.update(parsed["predictions"])
                    print(f"   {label} batch {batch_i+1}/{len(batches)}: {count} genes")
            except Exception as e:
                print(f"   {label} batch {batch_i+1} failed: {e}")

        # Cleanup: strip None probability values to prevent downstream crashes


        for sym in list(predictions.keys()):
            pred = predictions.get(sym)
            if isinstance(pred, dict):
                if pred.get("probability") is None:
                    pred["probability"] = 0.0
                if pred.get("confidence") is None or pred.get("confidence") == "":
                    pred["confidence"] = "LOW"

        return predictions

    def _ensemble_llm_gps(
        self, llm_preds: dict, candidate_genes: list,
    ) -> dict:
        """Smart ensemble: LLM-dominant, GPS-augmented where LLM is uncertain.

        Strategy:
          - LLM confident (prob ≥ 0.8 or ≤ 0.2): ~90% LLM + ~10% GPS
          - LLM uncertain (prob ≈ 0.5): 50% LLM + 50% GPS
          - Smooth linear transition in between
          - GPS weight never exceeds 50% — LLM always has the final say

        Formula:
          confidence = |llm_prob - 0.5| * 2          # 0→1
          alpha      = 0.5 + 0.4 * confidence        # LLM weight: 0.5→0.9
          final      = alpha * llm_prob + (1-alpha) * gps_score
        """
        ensemble = {}
        for g in candidate_genes:
            sym = g["gene_name"]
            # GPS score (from GeneDataAgent / JSON predictions)


            gps_score = g.get("gps_score")
            if gps_score is None:
                gps_score = 0.5

            llm_prob = 0.5
            if sym in llm_preds and isinstance(llm_preds[sym], dict):
                raw = llm_preds[sym].get("probability")
                llm_prob = raw if raw is not None else 0.5

            llm_confidence = abs(llm_prob - 0.5) * 2
            alpha = 0.5 + 0.4 * llm_confidence

            ensemble_prob = round(alpha * llm_prob + (1 - alpha) * gps_score, 4)
            ensemble[sym] = {
                "probability": ensemble_prob,
                "llm_prob": round(llm_prob, 4),
                "gps_score": round(gps_score, 4),
                "llm_weight": round(alpha, 2),
                "rationale": (
                    f"LLM confidence={llm_confidence:.2f} → "
                    f"weight {alpha:.0%} LLM + {1-alpha:.0%} GPS"
                ),
            }

        return ensemble

    def _parse_candidates(self, data_context: str) -> tuple[list, dict]:
        """Extract candidate gene list and selection metadata from data_context."""
        if not data_context:
            return [], {}
        try:
            ctx = json.loads(data_context) if isinstance(data_context, str) else data_context
            genes = ctx.get("candidate_genes", [])
            meta = ctx.get("selection_meta", {})
            if genes:
                return genes, meta
        except (json.JSONDecodeError, TypeError):
            pass
        return [], {}

    async def _generate_gene_list(self, goal: str, plan_context: dict) -> list:
        """llm_only mode: LLM generates candidate gene list from training knowledge."""
        disease = plan_context.get("disease_context", "disease")
        prompt = f"""Based on your training knowledge, list 30 genes strongly associated with {disease}.

Research goal: {goal}

Return ONLY JSON:
{{"genes": [
  {{"gene_name": "SYMBOL", "rationale": "brief reason for inclusion"}},
  ...
]}}

Include a mix of: well-established disease genes, emerging candidates, and functionally related genes.
Output ONLY the JSON array — no other text."""

        try:
            result = await self.autonomous_thinking(
                prompt=prompt, state=None,
                system_message="You are a genomics expert. Output ONLY JSON.",
                max_iterations=1,
            )
            parsed = self._extract_json_block(result.get("response", ""), debug_label="gene_list")
            if isinstance(parsed, dict) and "genes" in parsed:
                genes = parsed["genes"]
                # convert to candidate_genes format


                return [
                    {
                        "gene_name": g.get("gene_name", "?"),
                        "gene_degree": None,
                        "degree_tier": "llm_generated",
                        "top_neighbors": [],
                        "gps_score": None,
                        "ground_truth": None,
                        "source": "llm_knowledge",
                    }
                    for g in genes if g.get("gene_name")
                ]
        except Exception:
            pass

        # fallback: empty


        return []

    def _extract_json_block(self, text: str, debug_label: str = "") -> Any:
        import re
        try:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                return json.loads(match.group(1))
            # Try to find a complete JSON object


            start = text.find("{")
            if start != -1:
                # Find the matching closing brace


                depth = 0
                for i, ch in enumerate(text[start:], start):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:i + 1]
                            return json.loads(candidate)
        except json.JSONDecodeError as e:
            if debug_label:
                preview = text[:300].replace("\n", "\\n")
                print(f"   ️ [{debug_label}] JSON parse error: {e} | preview: {preview}")
        except Exception:
            pass
        if debug_label:
            preview = text[:200].replace("\n", "\\n")
            print(f"   ️ [{debug_label}] No JSON found | preview: {preview}")
        return None

    def _load_gene_names(self, dataset_name: str) -> dict:
        try:
            import h5py, os
            h5_dir = os.environ.get("CROSSGENE_H5_DIR", r"D:\pyProject\GPS\dataset\networks")
            h5_path = f'{h5_dir}/{dataset_name}_multiomics.h5'
            with h5py.File(h5_path, 'r') as f:
                if 'gene_names' not in f:
                    return {}
                names = f['gene_names'][:]
                result = {}
                for i, row in enumerate(names):
                    symbol = row[1]
                    if isinstance(symbol, bytes):
                        symbol = symbol.decode('utf-8')
                    result[i] = symbol
                return result
        except Exception:
            return {}

    def _get_test_genes(self) -> list:
        """Get a few test genes with ground truth labels for calibration."""
        try:
            import h5py, numpy as np, os
            h5_path = os.environ.get("CROSSGENE_H5_DIR", r"D:\pyProject\GPS\dataset\networks") + "/CPDB_multiomics.h5"
            with h5py.File(h5_path, 'r') as f:
                y_test = f['y_test'][:]
                mask_test = f['mask_test'][:]
                gene_names_raw = f['gene_names'][:] if 'gene_names' in f else None
                test_ids = np.where(mask_test == 1)[0]
                samples = []
                for tid in test_ids[:10]:
                    label = int(y_test[tid][0])
                    name = ""
                    if gene_names_raw is not None and tid < len(gene_names_raw):
                        row = gene_names_raw[tid]
                        s = row[1]
                        if isinstance(s, bytes):
                            s = s.decode('utf-8')
                        name = s
                    samples.append({"gene_id": int(tid), "gene_name": name, "label": label})
                return samples
        except Exception:
            return []
