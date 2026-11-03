#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, json, asyncio, time, random
import numpy as np

os.environ.pop("CROSSGENE_QUICK", None)
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from core.config import Settings
from agents.llm_fab_agent import LLMFabAgent
from agents.gene_data_agent import GeneDataAgent
from tools.h5_reader import select_candidate_genes

GOAL = "寻找与肝癌相关的基因"
SEED = 42


async def main():
    random.seed(SEED)
    np.random.seed(SEED)
    settings = Settings()

    # 加载全部 796 基因


    selection = select_candidate_genes("CPDB",
        n_test_pos=None, n_test_neg=None, n_hub=50)
    genes = selection.get("candidates", [])
    print(f"Loaded: {len(genes)} genes "
          f"({sum(1 for g in genes if g.get('ground_truth')==1)}+, "
          f"{sum(1 for g in genes if g.get('ground_truth')==0)}-, "
          f"{sum(1 for g in genes if g.get('source')=='hub_discovery')} hub)")

    # 加载 GPS 分数


    da = GeneDataAgent(settings=settings)
    gps_scores = da._load_gps_scores("CPDB")
    for g in genes:
        name = g.get("gene_name", "")
        if name in gps_scores:
            g["gps_score"] = round(gps_scores[name], 4)
    n_gps = sum(1 for g in genes if g.get("gps_score") is not None)
    print(f"GPS scores: {n_gps}/{len(genes)}")

    # 获取 LLM 预测


    agent = LLMFabAgent(settings=settings)
    await agent._initialize()

    state = {
        "iteration": 1, "goal": GOAL, "mode": "network_enhanced",
        "candidate_genes": genes, "data_context": json.dumps({"n_genes": len(genes)}),
        "feedback": "", "plan": {"disease_context": "liver cancer"},
    }

    t0 = time.time()
    result = await agent.run(state)
    fab = result.get("fab_results", {})
    llm_only_preds = fab.get("baseline_llm_only", {})
    ensemble_preds = fab.get("proposed_ensemble", {})
    t_llm = time.time() - t0
    print(f"LLM predictions: {len(llm_only_preds)} genes in {t_llm:.0f}s")

    # 组 1: 纯 GPS（只用 gps_score）


    gps_scores_list = []
    gps_labels = []
    for g in genes:
        ts = g.get("gps_score")
        gt = g.get("ground_truth")
        if ts is not None and gt is not None:
            gps_scores_list.append(ts)
            gps_labels.append(gt)
    gps_auc, gps_aupr, gps_f1 = compute_metrics(gps_scores_list, gps_labels)
    n_gps_labeled = len(gps_scores_list)

    # 组 2: 纯 LLM（只用 llm_prob，从 baseline_llm_only 提取）


    llm_scores = []
    llm_labels = []
    for sym, pred in llm_only_preds.items():
        if not isinstance(pred, dict):
            continue
        prob = pred.get("probability")
        gt = pred.get("ground_truth")
        if prob is not None and gt is not None:
            llm_scores.append(prob)
            llm_labels.append(gt)
    llm_auc, llm_aupr, llm_f1 = compute_metrics(llm_scores, llm_labels)
    n_llm_labeled = len(llm_scores)

    # 组 3: LLM+GPS 融合（ensemble_preds）


    ens_scores = []
    ens_labels = []
    for sym, pred in ensemble_preds.items():
        if not isinstance(pred, dict):
            continue
        prob = pred.get("probability")
        gt = pred.get("ground_truth")
        if prob is not None and gt is not None:
            ens_scores.append(prob)
            ens_labels.append(gt)
    ens_auc, ens_aupr, ens_f1 = compute_metrics(ens_scores, ens_labels)
    n_ens_labeled = len(ens_scores)

    # 组 4: 简单平均 (LLM+GPS 各 50%)


    avg_scores = []
    avg_labels = []
    for sym in set(llm_only_preds.keys()) & set(ensemble_preds.keys()):
        llm_p = llm_only_preds.get(sym, {})
        ens_p = ensemble_preds.get(sym, {})
        if not isinstance(llm_p, dict) or not isinstance(ens_p, dict):
            continue
        llm_prob = llm_p.get("probability")
        gps_score = ens_p.get("gps_score")
        gt = llm_p.get("ground_truth")
        if llm_prob is not None and gps_score is not None and gt is not None:
            avg_scores.append(0.5 * llm_prob + 0.5 * gps_score)
            avg_labels.append(gt)
    avg_auc, avg_aupr, avg_f1 = compute_metrics(avg_scores, avg_labels)

    # 输出


    print(f"\n{'='*80}")
    print("核心消融: LLM vs GPS vs Ensemble")
    print(f"有标签基因数: GPS={n_gps_labeled}, LLM={n_llm_labeled}, Ensemble={n_ens_labeled}, Avg={len(avg_scores)}")
    print(f"{'='*80}")

    h = f"{'Strategy':<25s} {'AUC':>8s} {'AUPR':>8s} {'F1':>8s}"
    print(h)
    print("-" * 50)

    baseline = {"auc": llm_auc, "aupr": llm_aupr, "f1": llm_f1}
    for name, auc, aupr, f1 in [
        ("纯 GPS (Graphormer)", gps_auc, gps_aupr, gps_f1),
        ("纯 LLM (训练知识)", llm_auc, llm_aupr, llm_f1),
        ("LLM+GPS 简单平均 (50/50)", avg_auc, avg_aupr, avg_f1),
        ("LLM+GPS 智能融合 (ensemble)", ens_auc, ens_aupr, ens_f1),
    ]:
        d_auc = auc - baseline["auc"] if baseline["auc"] else 0
        d_aupr = aupr - baseline["aupr"] if baseline["aupr"] else 0
        print(f"{name:<25s} {auc:.4f}   {aupr:.4f}   {f1:.4f}     ΔAUC={d_auc:+.4f} ΔAUPR={d_aupr:+.4f}")

    # 哪些基因从融合中获益？


    improved = 0
    degraded = 0
    for sym in set(llm_only_preds.keys()) & set(ensemble_preds.keys()):
        llm_p = llm_only_preds.get(sym, {})
        ens_p = ensemble_preds.get(sym, {})
        if not isinstance(llm_p, dict) or not isinstance(ens_p, dict):
            continue
        gt = llm_p.get("ground_truth")
        llm_prob = llm_p.get("probability")
        ens_prob = ens_p.get("probability")
        if gt is not None and llm_prob is not None and ens_prob is not None:
            llm_err = abs(llm_prob - gt)
            ens_err = abs(ens_prob - gt)
            if ens_err < llm_err:
                improved += 1
            elif ens_err > llm_err:
                degraded += 1

    print(f"\nFusion effect: {improved} genes improved vs pure LLM, {degraded} degraded")

    # 分正负样本看


    for label_name, label_val in [("正样本 (disease)", 1), ("负样本 (non-disease)", 0)]:
        t_s, t_l = zip(*[(s, l) for s, l in zip(gps_scores_list, gps_labels) if l == label_val])
        l_s, l_l = zip(*[(s, l) for s, l in zip(llm_scores, llm_labels) if l == label_val])
        e_s, e_l = zip(*[(s, l) for s, l in zip(ens_scores, ens_labels) if l == label_val])
        t_auc, _, _ = compute_metrics(list(t_s), list(t_l))
        l_auc, _, _ = compute_metrics(list(l_s), list(l_l))
        e_auc, _, _ = compute_metrics(list(e_s), list(e_l))
        print(f"\n{label_name} ({len(t_l)} genes):")
        print(f"  纯 GPS AUC={t_auc:.4f}  纯 LLM AUC={l_auc:.4f}  Ensemble AUC={e_auc:.4f}")

    await agent._shutdown()


def compute_metrics(scores, labels):
    import numpy as np
    n = len(labels)
    if n < 2:
        return None, None, None
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
        auc = round(float(roc_auc_score(labels_arr, scores_arr)), 4)
        aupr = round(float(average_precision_score(labels_arr, scores_arr)), 4)
        preds = (scores_arr >= 0.5).astype(int)
        f1 = round(float(f1_score(labels_arr, preds)), 4)
        return auc, aupr, f1
    except ImportError:
        pos = scores_arr[labels_arr == 1]
        neg = scores_arr[labels_arr == 0]
        if len(pos) > 0 and len(neg) > 0:
            n_correct = int(np.sum(pos[:, None] > neg[None, :]))
            n_ties = int(np.sum(pos[:, None] == neg[None, :]))
            auc = round((n_correct + 0.5 * n_ties) / (len(pos) * len(neg)), 4)
        else:
            auc = None
        tp = int(np.sum((scores_arr >= 0.5) & (labels_arr == 1)))
        fp = int(np.sum((scores_arr >= 0.5) & (labels_arr == 0)))
        fn = int(np.sum((scores_arr < 0.5) & (labels_arr == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0
        return auc, None, f1


if __name__ == "__main__":
    asyncio.run(main())
