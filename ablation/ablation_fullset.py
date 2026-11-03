#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import os, sys, json, asyncio, time, random
import numpy as np

os.environ.pop("CROSSGENE_QUICK", None)
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from workflow.orchestrator import GeneAgentOrchestrator
from core.config import Settings
from tools.h5_reader import select_candidate_genes
from agents.gene_data_agent import GeneDataAgent
from tools.online_db import batch_validate_genes, _clean_symbol as clean_sym

GOAL = "寻找与肝癌相关的基因"
SEED = 42

CONDITIONS = {
    "full":                "完整系统(LLM+GPS)",
    "no_debate":           "No Debate",
    "no_dynamic_weight":   "No Dynamic Weight",
    "no_network":          "No Network",
    "no_literature":       "No Literature",
}
LABELS = CONDITIONS


async def run_one(ablation: str) -> dict:
    random.seed(SEED)
    np.random.seed(SEED)

    settings = Settings()
    orch = GeneAgentOrchestrator(settings=settings)
    await orch.initialize()

    from workflow.state import create_initial_state
    state = create_initial_state(GOAL, max_iterations=1)
    state["plan"] = {"mode": "network_enhanced", "dataset": "CPDB",
                     "disease_context": "liver cancer",
                     "gene_categories": ["all"]}
    state["mode"] = "network_enhanced"
    state["disease_context"] = "liver cancer"

    try:
        # 加载全部 test 基因 + 50 hub


        selection = select_candidate_genes("CPDB",
            n_test_pos=None,   # ALL 199 positive
            n_test_neg=None,   # ALL 547 negative
            n_hub=50)          # top 50 hub discovery
        state["candidate_genes"] = selection.get("candidates", [])
        n_genes = len(state["candidate_genes"])

        # 附加 GPS 分数（No Network 清除）


        da = GeneDataAgent(settings=settings)
        gps_scores = da._load_gps_scores("CPDB")
        for g in state["candidate_genes"]:
            name = g.get("gene_name", "")
            if name in gps_scores:
                g["gps_score"] = round(gps_scores[name], 4)

        if ablation == "no_network":
            for g in state["candidate_genes"]:
                g["gps_score"] = None
                g["gene_degree"] = None
                g["degree_tier"] = "no_network"
                g["top_neighbors"] = []

        n_pos = sum(1 for g in state["candidate_genes"] if g.get("ground_truth") == 1)
        n_neg = sum(1 for g in state["candidate_genes"] if g.get("ground_truth") == 0)
        n_hub = sum(1 for g in state["candidate_genes"] if g.get("source") == "hub_discovery")
        state["data_context"] = json.dumps({
            "n_genes": n_genes, "n_positive": n_pos, "n_negative": n_neg, "n_hub": n_hub,
        })
        print(f"   Loaded: {n_genes} genes ({n_pos}+, {n_neg}-, {n_hub} hub)")

        state["iteration"] = 1

        # Phase 1: LLMFabAgent 预测全部基因


        t0 = time.time()
        knowledge_result = await orch.agents["knowledge"].run(dict(state))
        state.update(knowledge_result)
        fab = state.get("fab_results", {})
        predictions = fab.get("predictions", {}) if isinstance(fab, dict) else {}
        baseline_preds = fab.get("baseline_llm_only", {}) if isinstance(fab, dict) else {}
        proposed_preds = fab.get("proposed_ensemble", {}) if isinstance(fab, dict) else {}
        t_llm = time.time() - t0

        # Phase 2: PubMed 查询 top 100 基因（分层：top 40 + mid 30 + bot 30）


        pubmed_evidence = {}
        if ablation != "no_literature":
            # 按 LLM 概率排序


            sorted_genes = sorted(
                [(s, p) for s, p in predictions.items()
                 if isinstance(p, dict)],
                key=lambda x: x[1].get("probability") or 0, reverse=True,
            )
            n_sorted = len(sorted_genes)
            top40 = [s for s, _ in sorted_genes[:40]]
            mid_start = max(0, n_sorted // 2 - 15)
            mid30 = [s for s, _ in sorted_genes[mid_start:mid_start+30]]
            bot_start = max(0, n_sorted - 30)
            bot30 = [s for s, _ in sorted_genes[bot_start:]]
            query_genes = list(dict.fromkeys(top40 + mid30 + bot30))

            print(f"   PubMed: querying {len(query_genes)} genes (stratified)...")
            t_pub = time.time()
            clean = [clean_sym(s) for s in query_genes
                    if not s.startswith("ENSG") and not s.startswith("Gene_")]
            for i in range(0, len(clean), 10):
                batch = clean[i:i+10]
                result = batch_validate_genes(batch, "liver cancer")
                if result.get("status") == "success":
                    for sym, info in result.get("per_gene", {}).items():
                        if isinstance(info, dict):
                            p = info.get("pubmed", {})
                            if isinstance(p, dict):
                                pubmed_evidence[sym] = p.get("pubmed_count", 0)
                if i + 10 < len(clean):
                    await asyncio.sleep(0.4)

            n_high = sum(1 for c in pubmed_evidence.values() if c > 100)
            n_med = sum(1 for c in pubmed_evidence.values() if 10 <= c <= 100)
            n_low = sum(1 for c in pubmed_evidence.values() if c < 10)
            print(f"   PubMed done: {len(pubmed_evidence)} genes "
                  f"({n_high} high, {n_med} med, {n_low} low) "
                  f"in {time.time()-t_pub:.0f}s")
        else:
            print("   PubMed: skipped (No Literature)")

        state["pubmed_evidence"] = pubmed_evidence
        state["literature_report"] = json.dumps({"pubmed_evidence": pubmed_evidence})

        # Phase 3: Analysis


        analysis_result = await orch.agents["analysis"].run(dict(state))
        state.update(analysis_result)

        # Phase 4: Debate / No Debate / Fixed Weight


        if ablation == "no_debate":
            state["debate_resolutions"] = {}
        elif ablation == "no_dynamic_weight":
            resolutions = {}
            for sym in predictions:
                if not isinstance(predictions[sym], dict):
                    continue
                p = predictions[sym]
                llm = p.get("llm_prob") or p.get("probability") or 0.5
                gps = p.get("gps_score") or 0.5
                resolutions[sym] = {
                    "gene": sym, "status": "resolved",
                    "final_probability": round(0.5 * llm + 0.5 * gps, 4),
                    "llm_weight": 0.5, "gps_weight": 0.5,
                }
            state["debate_resolutions"] = resolutions
            state["stalemate_genes"] = []
        else:
            # 运行辩论（仅完整系统 和 No Network/No Literature）


            state = await orch._run_debates(state)

        # 收集指标


        analysis = {}
        try:
            analysis = json.loads(state.get("analysis_report", "{}"))
        except Exception:
            pass

        ev = analysis.get("evaluation", {})
        baseline_ev = ev.get("baseline_llm_only", {})
        proposed_ev = ev.get("proposed_ensemble", {})
        alignment = analysis.get("alignment_summary", {})
        debate_res = state.get("debate_resolutions", {})

        # 权重统计


        weights = [r["llm_weight"] for r in debate_res.values()
                  if isinstance(r, dict) and r.get("llm_weight") is not None]

        # PubMed 分层一致率


        n_high_pubmed = sum(1 for c in pubmed_evidence.values() if c > 100)
        n_queried = len(pubmed_evidence)

        # LLM-GPS 分歧


        disagreements = []
        for sym, p in predictions.items():
            if not isinstance(p, dict):
                continue
            llm_p = p.get("llm_prob") or p.get("probability")
            gps_s = p.get("gps_score")
            if llm_p is not None and gps_s is not None:
                gap = abs(llm_p - gps_s)
                if gap > 0.3:
                    disagreements.append({"gene": sym, "gap": round(gap, 4)})

        n_discordant = sum(
            1 for g in analysis.get("per_gene_analysis", [])
            if g.get("alignment") in ("llm_high_db_low", "llm_low_db_high")
        )

        return {
            "ablation": ablation,
            "label": LABELS[ablation],
            # 数据规模


            "n_genes": n_genes,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_hub": n_hub,
            "n_labeled": proposed_ev.get("n_labeled", 0),
            # Baseline (LLM only)


            "baseline_auc": baseline_ev.get("auc"),
            "baseline_aupr": baseline_ev.get("aupr"),
            "baseline_f1": baseline_ev.get("f1"),
            "baseline_precision": baseline_ev.get("precision"),
            "baseline_recall": baseline_ev.get("recall"),
            "baseline_mean_pos": baseline_ev.get("mean_prob_positive"),
            "baseline_mean_neg": baseline_ev.get("mean_prob_negative"),
            # Proposed (LLM+GPS)


            "proposed_auc": proposed_ev.get("auc"),
            "proposed_aupr": proposed_ev.get("aupr"),
            "proposed_f1": proposed_ev.get("f1"),
            "proposed_precision": proposed_ev.get("precision"),
            "proposed_recall": proposed_ev.get("recall"),
            # PubMed


            "pubmed_queried": n_queried,
            "pubmed_high": n_high_pubmed,
            "pubmed_concordance": alignment.get("concordance_rate", 0),
            "n_concordant": alignment.get("concordant", 0),
            "n_discordant": n_discordant,
            "pubmed_breakdown": alignment.get("breakdown", {}),
            # 辩论/权重


            "n_disputed": len(debate_res),
            "n_disagreements": len(disagreements),
            "mean_llm_weight": round(float(np.mean(weights)), 4) if weights else None,
            "std_llm_weight": round(float(np.std(weights)), 4) if weights else None,
            # 耗时


            "t_llm_sec": round(t_llm, 0),
            "error": None,
        }
    except Exception as e:
        import traceback
        return {"ablation": ablation, "label": LABELS[ablation],
                "error": str(e), "traceback": traceback.format_exc()[-500:]}
    finally:
        await orch.shutdown()


async def main():
    print("=" * 75)
    print("全量消融实验 — CPDB 全部 test 基因 (746+)")
    print(f"Conditions: {list(LABELS.values())}")
    print("=" * 75)

    all_results = []
    for ab in CONDITIONS:
        label = LABELS[ab]
        print(f"\n{'─'*55}")
        print(f" {label}")
        print(f"{'─'*55}")
        t0 = time.time()
        result = await run_one(ab)
        elapsed_min = round((time.time() - t0) / 60, 1)
        result["elapsed_min"] = elapsed_min

        if result.get("error"):
            print(f"  ❌ FAILED: {result['error'][:120]}")
        else:
            print(f"\n   {result['n_genes']} genes, "
                  f"LLM AUC={result['baseline_auc']:.4f}, "
                  f"Ensemble AUC={result['proposed_auc']:.4f}, "
                  f"PubMed一致={result['pubmed_concordance']:.0%}, "
                  f"分歧={result['n_disagreements']}, "
                  f"不一致={result['n_discordant']}")
            print(f"  ⏱️  {elapsed_min:.0f}min "
                  f"(LLM={result.get('t_llm_sec', 0):.0f}s)")
        all_results.append(result)

    ok = [r for r in all_results if not r.get("error")]
    full = next((r for r in ok if r["ablation"] == "full"), None)

    # Table 1: Baseline (LLM only)


    print(f"\n{'='*110}")
    print("Table 1: Baseline (LLM only) — ALL {n_genes} genes".format(
        n_genes=full["n_genes"] if full else "?"))
    print(f"{'='*110}")
    h = (f"{'Condition':<22s} {'AUC':>8s} {'AUPR':>8s} {'F1':>8s} "
         f"{'Prec':>8s} {'Rec':>8s} {'Mean_pos':>10s} {'Mean_neg':>10s}")
    print(h)
    print("-" * 110)
    for r in ok:
        d = "" if r["ablation"] == "full" or not full else \
            f"  ΔAUC={r['baseline_auc'] - full['baseline_auc']:+.4f}"
        print(f"{r['label']:<22s} "
              f"{r['baseline_auc']:.4f}  {r['baseline_aupr']:.4f}  "
              f"{r['baseline_f1']:.4f}  {r['baseline_precision']:.4f}  "
              f"{r['baseline_recall']:.4f}  "
              f"{r['baseline_mean_pos'] or 'N/A':>10s}  "
              f"{r['baseline_mean_neg'] or 'N/A':>10s}"
              f"{d}")

    # Table 2: Proposed (LLM+GPS)


    print(f"\n{'='*110}")
    print("Table 2: Proposed (LLM + GPS) — ALL genes")
    print(f"{'='*110}")
    h2 = (f"{'Condition':<22s} {'AUC':>8s} {'AUPR':>8s} {'F1':>8s} "
          f"{'Prec':>8s} {'Rec':>8s} {'Δvs Baseline':>14s}")
    print(h2)
    print("-" * 110)
    for r in ok:
        delta_auc = (r["proposed_auc"] or 0) - (r["baseline_auc"] or 0)
        delta_f1 = (r["proposed_f1"] or 0) - (r["baseline_f1"] or 0)
        delta_str = f"ΔAUC={delta_auc:+.4f} ΔF1={delta_f1:+.4f}"
        d2 = "" if r["ablation"] == "full" or not full else \
            f"  vs_full_ΔAUC={r['proposed_auc'] - full['proposed_auc']:+.4f}"
        print(f"{r['label']:<22s} "
              f"{r['proposed_auc']:.4f}  {r['proposed_aupr']:.4f}  "
              f"{r['proposed_f1']:.4f}  {r['proposed_precision']:.4f}  "
              f"{r['proposed_recall']:.4f}  "
              f"{delta_str:<14s}"
              f"{d2}")

    # Table 3: PubMed + Alignment + Weights


    print(f"\n{'='*110}")
    print("Table 3: PubMed 交叉验证 + LLM-GPS 分歧 + 权重分布")
    print(f"{'='*110}")
    h3 = (f"{'Condition':<22s} {'PubMed查':>7s} {'高证':>4s} "
          f"{'一致率':>7s} {'不一致':>6s} {'分歧':>5s} "
          f"{'权μ':>6s} {'权σ':>6s}")
    print(h3)
    print("-" * 110)
    for r in ok:
        mw = f"{r['mean_llm_weight']:.3f}" if r['mean_llm_weight'] is not None else "N/A"
        sw = f"{r['std_llm_weight']:.3f}" if r['std_llm_weight'] is not None else "N/A"
        print(f"{r['label']:<22s} "
              f"{r['pubmed_queried']:>7d} {r['pubmed_high']:>4d} "
              f"{r['pubmed_concordance']:>6.0%} {r['n_discordant']:>6d} "
              f"{r['n_disagreements']:>5d} "
              f"{mw:>6s} {sw:>6s}")

    with open("ablation_fullset_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n📁 Saved to ablation_fullset_results.json")


if __name__ == "__main__":
    asyncio.run(main())
