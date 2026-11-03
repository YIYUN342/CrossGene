#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import os, sys, json, asyncio, time, random
import numpy as np
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

# ️ NOT quick mode — full dataset (~750 genes)


os.environ.pop("CROSSGENE_QUICK", None)

from workflow.orchestrator import GeneAgentOrchestrator
from core.config import Settings

GOAL = "寻找与肝癌相关的基因"
ABLATIONS = [None, "no_critic", "no_mediator", "no_memory"]
LABELS = {None: "完整系统", "no_critic": "-CriticAgent",
          "no_mediator": "-Mediator", "no_memory": "-MemoryAgent"}
SEEDS = [42, 123, 456]
RESULTS_FILE = "ablation_full_results.json"


async def run_one(ablation: str | None, goal: str, seed: int) -> dict:
    """Run one experiment with locked CPDB and fixed seed."""
    random.seed(seed)
    np.random.seed(seed)

    settings = Settings()
    orch = GeneAgentOrchestrator(settings=settings, ablation=ablation)

    try:
        await orch.initialize()

        from workflow.state import create_initial_state
        state = create_initial_state(goal, max_iterations=1)
        state["plan"] = {
            "mode": "network_enhanced", "dataset": "CPDB",
            "disease_context": "liver cancer",
            "gene_categories": ["oncogenes", "tumor suppressor genes"],
            "reason": f"ABLATION: {ablation}, seed={seed}",
        }
        state["mode"] = "network_enhanced"
        state["disease_context"] = "liver cancer"
        state["iteration"] = 1

        # Phase 2: Data


        data_result = await orch.agents["data"].run(dict(state))
        state.update(data_result)

        # Phase 3: Predict


        knowledge_result = await orch.agents["knowledge"].run(dict(state))
        state.update(knowledge_result)

        # Phase 4: Validate → Analyze


        lit_result = await orch.agents["literature"].run(dict(state))
        state.update(lit_result)
        analysis_result = await orch.agents["analysis"].run(dict(state))
        state.update(analysis_result)

        # Phase 5: Debate + memory + conclusion


        if state.get("mode") == "network_enhanced":
            state = await orch._run_debates(state)
        if ablation != "no_memory":
            mem_result = await orch.agents["memory"].run(dict(state))
            state.update(mem_result)

        # Extract metrics


        fab = state.get("fab_results", {})
        predictions = fab.get("predictions", {}) if isinstance(fab, dict) else {}
        analysis = {}
        try:
            analysis = json.loads(state.get("analysis_report", "{}"))
        except Exception:
            pass
        ev = analysis.get("evaluation", {})
        proposed = ev.get("proposed_ensemble", {})
        alignment = analysis.get("alignment_summary", {})

        debate_res = state.get("debate_resolutions", {})
        n_discordant = sum(
            1 for g in analysis.get("per_gene_analysis", [])
            if g.get("alignment") in ("llm_high_db_low", "llm_low_db_high")
        )

        return {
            "ablation": ablation, "label": LABELS.get(ablation, "?"),
            "seed": seed,
            "n_genes": len(state.get("candidate_genes", [])),
            "n_predicted": len(predictions),
            "n_labeled": proposed.get("n_labeled", 0),
            "auc": proposed.get("auc"),
            "aupr": proposed.get("aupr"),
            "f1": proposed.get("f1"),
            "precision": proposed.get("precision"),
            "recall": proposed.get("recall"),
            "n_disputed": len(debate_res),
            "n_resolved": sum(1 for r in debate_res.values() if r.get("status") == "resolved"),
            "n_stalemate": sum(1 for r in debate_res.values() if r.get("status") == "stalemate"),
            "pubmed_concordance": alignment.get("concordance_rate", 0),
            "n_concordant": alignment.get("concordant", 0),
            "n_discordant": n_discordant,
            "n_high_evidence": sum(1 for c in state.get("pubmed_evidence", {}).values() if c > 100),
            "error": None,
        }
    except Exception as e:
        import traceback
        return {"ablation": ablation, "label": LABELS.get(ablation, "?"),
                "seed": seed, "error": str(e), "traceback": traceback.format_exc()}
    finally:
        await orch.shutdown()


def compute_stats(metrics_list: list[dict], key: str) -> tuple:
    """Compute mean ± std for a metric across runs."""
    vals = [m[key] for m in metrics_list if m.get(key) is not None and not m.get("error")]
    if len(vals) < 2:
        return None, None, len(vals)
    return np.mean(vals), np.std(vals), len(vals)


async def main():
    print("=" * 75)
    print("PROPER ABLATION STUDY")
    print(f"Goal: {GOAL}")
    print(f"Dataset: CPDB (locked)")
    print(f"Mode: FULL ({'NOT QUICK'})")
    print(f"Conditions: {len(ABLATIONS)}, Repeats: {len(SEEDS)}")
    print(f"Total runs: {len(ABLATIONS) * len(SEEDS)}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)

    all_results: list[dict] = []

    # Load existing partial results (resume support)


    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            completed = {(r["ablation"], r["seed"]) for r in all_results if not r.get("error")}
            print(f"Resuming from {len(all_results)} existing results "
                  f"({len(completed)} successful)")
        except Exception:
            pass

    total = len(ABLATIONS) * len(SEEDS)
    run_idx = len([r for r in all_results if not r.get("error")])

    for abl in ABLATIONS:
        for seed in SEEDS:
            # Skip already completed runs


            if any(r.get("ablation") == abl and r.get("seed") == seed
                   and not r.get("error") for r in all_results):
                run_idx += 1
                continue

            label = LABELS[abl]
            run_idx += 1
            print(f"\n{'='*65}")
            print(f"[{run_idx}/{total}] {label} (seed={seed})")
            print(f"{'='*65}")

            t0 = time.time()
            result = await run_one(abl, GOAL, seed)
            elapsed = time.time() - t0
            result["elapsed_min"] = round(elapsed / 60, 1)

            if result.get("error"):
                print(f"  ❌ FAILED: {result['error'][:100]}")
            else:
                print(f"   AUC={result['auc']:.4f}, F1={result['f1']:.4f}, "
                      f"AUPR={result['aupr']:.4f}, 争议={result['n_disputed']}, "
                      f"一致率={result['pubmed_concordance']:.1%}, "
                      f"{result['elapsed_min']:.0f}min")

            all_results.append(result)

            # Save after each run (crash-safe)


            with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # Statistical analysis


    print(f"\n\n{'='*75}")
    print("ABLATION STUDY — STATISTICAL ANALYSIS")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*75}")

    # Filter successful runs


    ok = [r for r in all_results if not r.get("error")]
    n_ok = len(ok)
    print(f"Successful runs: {n_ok}/{len(all_results)}")

    # Per-condition stats


    metrics = ["auc", "aupr", "f1", "precision", "recall",
               "pubmed_concordance", "n_disputed", "n_discordant"]

    print(f"\n{'Condition':<20s}", end="")
    for m in metrics:
        print(f"  {m:<22s}", end="")
    print(f"\n{'-'*200}")

    baseline_stats = {}
    for abl in ABLATIONS:
        label = LABELS[abl]
        runs = [r for r in ok if r["ablation"] == abl]

        print(f"{label:<20s}", end="")
        for m in metrics:
            mean, std, n = compute_stats(runs, m)
            if mean is not None:
                if abl is None:
                    baseline_stats[m] = mean
                delta = ""
                if abl is not None and m in baseline_stats:
                    d = mean - baseline_stats[m]
                    if abs(d) > 0.001:
                        delta = f" Δ={d:+.4f}"
                print(f"  {mean:.4f}±{std:.4f}{delta:<10s}", end="")
            else:
                print(f"  {'N/A':<22s}", end="")
        print()

    # Per-gene overlap analysis


    print(f"\n{'='*75}")
    print("DEBATE QUALITY ANALYSIS")
    print(f"{'='*75}")
    print(f"{'Condition':<20s} {'争议基因(mean)':>10s} {'解决(mean)':>8s} {'僵局(mean)':>8s} {'PubMed不一致':>8s}")
    print("-" * 65)
    for abl in ABLATIONS:
        label = LABELS[abl]
        runs = [r for r in ok if r["ablation"] == abl]
        n_dis = compute_stats(runs, "n_disputed")[0] or 0
        n_res = compute_stats(runs, "n_resolved")[0] or 0
        n_sta = compute_stats(runs, "n_stalemate")[0] or 0
        n_dsc = compute_stats(runs, "n_discordant")[0] or 0
        print(f"{label:<20s} {n_dis:>10.1f} {n_res:>8.1f} {n_sta:>8.1f} {n_dsc:>8.1f}")

    print(f"\nFull results: {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
