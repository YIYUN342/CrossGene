#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CrossGene — LLM + GPS 多智能体基因-疾病关联预测系统入口。

用法:
  python run.py
  python run.py --dataset CPDB --goal "Find breast cancer related genes"
  python run.py --max-iter 5
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime

# Ensure cwd is project root


os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from workflow.graph import GeneWorkflow
from workflow.orchestrator import GeneAgentOrchestrator
from core.config import Settings


async def main(goal: str, max_iterations: int, use_debate: bool = False,
                ablation: str | None = None, dataset: str | None = None):
    print("=" * 70)
    label = "Debate-Driven Multi-Agent" if use_debate else "Iterative LLM-Based"
    print(f"CrossGene — {label} Gene-Disease Prediction")
    print(f"Goal: {goal}")
    print(f"Max iterations: {max_iterations}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    settings = Settings()
    print(f"LLM: {settings.llm.provider} / {settings.llm.model}")

    if use_debate:
        workflow = GeneAgentOrchestrator(settings=settings, ablation=ablation)
    else:
        workflow = GeneWorkflow(settings=settings)

    try:
        final_state = await workflow.run(goal, max_iterations=max_iterations,
                                         forced_dataset=dataset)

        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)

        # Iteration summary


        iteration_history = final_state.get("iteration_history", [])
        fab = final_state.get("fab_results", {})

        if isinstance(fab, dict):
            predictions = fab.get("predictions", {})
            baseline = fab.get("baseline_llm_only", {})
            proposed = fab.get("proposed_ensemble", {})

            print(f"\n[LLMFabAgent] Final Round ({fab.get('round_label', '?')})")
            print(f"  Baseline (LLM):   {len(baseline)} genes")
            print(f"  Proposed (LLM+GPS): {len(proposed)} genes")

            # Top 10


            items = sorted(
                predictions.items(),
                key=lambda x: (x[1].get("probability") or 0) if isinstance(x[1], dict) else 0,
                reverse=True,
            )[:10]
            print(f"\n  Top 10 Disease-Associated Gene Predictions:")
            print(f"  {'Gene':<10s} {'Ensemble':>8s}  {'LLM':>8s}  {'GPS':>8s}  {'PubMed':>8s}  {'Tier':<12s}")
            print(f"  {'-'*70}")
            for sym, pred in items:
                if isinstance(pred, dict):
                    llm_prob = baseline.get(sym, {}).get("probability") if isinstance(baseline.get(sym), dict) else None
                    gps = pred.get("gps_score", "N/A")
                    pubmed = pred.get("pubmed_count", pred.get("api_verification", "?"))
                    llm_str = f"{llm_prob:.4f}" if llm_prob is not None else "N/A"
                    gps_str = f"{gps:.4f}" if isinstance(gps, (int, float)) else str(gps)
                    pubmed_str = str(pubmed) if pubmed else "?"
                    print(f"  {sym:<10s} {pred.get('probability', 0):>8.4f}  {llm_str:>8s}  "
                          f"{gps_str:>8s}  {pubmed_str:>8s}  {pred.get('degree_tier', '?'):<12s}")

        # Analysis


        ar = final_state.get("analysis_report", "")
        try:
            analysis_data = json.loads(ar) if isinstance(ar, str) else ar
            if isinstance(analysis_data, dict):
                ev = analysis_data.get("evaluation", {})
                bm = ev.get("baseline_llm_only", {})
                pm = ev.get("proposed_ensemble", {})
                align = analysis_data.get("alignment_summary", {})
                print(f"\n[Analysis]")
                print(f"  Baseline (LLM):   AUC={bm.get('auc')}, F1={bm.get('f1')}")
                print(f"  Proposed (LLM+GPS): AUC={pm.get('auc')}, F1={pm.get('f1')}")
                print(f"  PubMed Alignment: {align.get('concordant', '?')}/{align.get('total_genes', '?')} "
                      f"concordant ({align.get('concordance_rate', 0)*100:.0f}%)")
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

        # Iteration history


        print(f"\n[Iterations] {len(iteration_history)} rounds completed:")
        for entry in iteration_history:
            ev = entry.get("evaluation", {}).get("proposed_ensemble", {})
            print(f"  Round {entry.get('iteration', '?')}: {entry.get('decision', '?')}  "
                  f"AUC={ev.get('auc', '?')}, F1={ev.get('f1', '?')}  "
                  f"({entry.get('reason', '')[:60]})")

        # Final conclusion


        fc = final_state.get("final_conclusion", "")
        if fc:
            print(f"\n[Final Conclusion] ({len(str(fc))} chars)")
            print(str(fc)[:1000])

        # Save results


        save = {}
        for k, v in final_state.items():
            try:
                json.dumps(v)
                save[k] = v
            except Exception:
                save[k] = str(v)[:500]
        with open("result.json", "w", encoding="utf-8") as f:
            json.dump(save, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nFull results saved to: result.json")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        await workflow.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrossGene — Multi-Agent Gene-Disease Prediction with GPS")
    parser.add_argument("--goal", type=str,
                        default="Identify disease-associated genes from the CPDB gene interaction network")
    parser.add_argument("--max-iter", type=int, default=3,
                        help="Maximum iteration rounds (default: 3)")
    parser.add_argument("--debate", action="store_true",
                        help="Use debate-driven multi-agent system (Phase 4+)")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["no_critic", "no_mediator", "no_memory"],
                        help="Ablation experiment: disable one component")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Force a specific dataset (e.g., CPDB, LTG, STRINGdb)")
    args = parser.parse_args()

    # Quick mode: fewer genes, 1 iteration


    if os.environ.get("CROSSGENE_QUICK"):
        args.max_iter = 1
        print("[QUICK MODE] 1 iteration with fewer genes")

    asyncio.run(main(args.goal, args.max_iter,
                        use_debate=args.debate, ablation=args.ablation,
                        dataset=args.dataset))
