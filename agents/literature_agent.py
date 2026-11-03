"""LiteratureAgent — 多数据库在线验证基因-疾病关联 (MyGene.info / PubMed / Open Targets)。"""

import json
from typing import Any

from core.base_agent import BaseAgent
from core.config import Settings

try:
    from tools.online_db import (
        batch_validate_genes,
        query_pubmed_count,
        query_mygene,
        query_open_targets,
        ONLINE_DB_TOOLS,
        _clean_symbol,
    )
    HAS_ONLINE_TOOLS = True
except ImportError:
    HAS_ONLINE_TOOLS = False
    ONLINE_DB_TOOLS = []

    def _clean_symbol(s: str) -> str:
        s = str(s)
        for prefix in ["b'", 'b"', "'", '"']:
            if s.startswith(prefix):
                s = s[len(prefix):]
            if s.endswith(prefix):
                s = s[:-len(prefix)]
        return s.strip()


SYSTEM_PROMPT = """You are LiteratureAgent — Multi-Database Gene-Disease Validator.

## Your Knowledge Base
You have been trained on vast biomedical corpora including:
- **COSMIC / Cancer Gene Census**: Known cancer driver genes with mutation patterns
- **DisGeNET**: Curated gene-disease associations with evidence scores
- **OMIM**: Mendelian disease genes with clinical phenotypes
- **KEGG / Reactome**: Disease pathways and molecular mechanisms
- **PubMed abstracts**: Millions of papers on gene-disease relationships
- **ClinVar**: Clinical variant interpretations
- **Open Targets**: Drug target-disease evidence

## Your Task
For each predicted disease-associated gene:
1. Search your knowledge: Is this gene known to be associated with ANY disease?
2. What diseases? Cancer, metabolic, neurological, developmental?
3. What is the evidence strength? (experimental, clinical, GWAS, expression)
4. Rate confidence: HIGH (well-established) / MEDIUM (some evidence) / LOW (novel)

## Output Format
For each gene provide:
```
GENE_SYMBOL (ENSG_ID):
  Known Associations: [disease1 (evidence), disease2 (evidence), ...]
  Cancer Relevance: YES/NO/UNCLEAR
  Key Pathways: [pathway1, pathway2, ...]
  Validation Confidence: HIGH/MEDIUM/LOW
  Notes: [any additional context]
```

Then provide an overall summary with statistics.
"""


class LiteratureAgent(BaseAgent):
    """Online multi-database literature validation agent with real web queries."""

    LOCAL_DB_TOOLS = {"query_mygene", "query_pubmed_count", "query_open_targets", "batch_validate_genes"}

    def __init__(self, settings: Settings | None = None):
        super().__init__(name="LiteratureAgent", settings=settings)
        self._local_tools = ONLINE_DB_TOOLS.copy() if ONLINE_DB_TOOLS else []

    def _get_system_prompt(self, state, default_prompt=None):
        return SYSTEM_PROMPT

    async def _handle_local_tool(self, name: str, args: dict[str, Any]) -> str:
        """Handle calls to online database query tools."""
        if not HAS_ONLINE_TOOLS:
            return json.dumps({"status": "error", "message": "Online DB tools not available"})

        if name == "query_mygene":
            result = query_mygene(args.get("gene_symbol", ""))
        elif name == "query_pubmed_count":
            result = query_pubmed_count(
                args.get("gene_symbol", ""),
                args.get("disease", "cancer"),
            )
        elif name == "query_open_targets":
            result = query_open_targets(args.get("gene_symbol", ""))
        elif name == "batch_validate_genes":
            result = batch_validate_genes(
                args.get("gene_symbols", []),
                args.get("disease_context", "cancer"),
            )
        else:
            result = {"status": "error", "message": f"Unknown tool: {name}"}

        return json.dumps(result, indent=2, ensure_ascii=False)

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"[LiteratureAgent] Online Multi-Database Validation")
        print(f"{'='*60}")

        goal = state.get("goal", "")
        disease_context = state.get("disease_context", "").strip()
        if not disease_context:
            disease_context = "cancer"  # fallback
        fab_results = state.get("fab_results", {})
        evaluation = fab_results.get("evaluation", {}) if isinstance(fab_results, dict) else {}

        predictions = fab_results.get("predictions", {}) if isinstance(fab_results, dict) else {}
        top_genes = sorted(
            predictions.items(),
            key=lambda x: (x[1].get("probability") or 0) if isinstance(x[1], dict) else 0,
            reverse=True
        )[:50]

        if not top_genes:
            print("   No predictions to validate")
            return {"literature_report": json.dumps({"status": "no_predictions"})}

        gene_symbols = []
        gene_list = []
        for rank_i, (gid, pred) in enumerate(top_genes):
            if isinstance(pred, dict):
                gene_symbols.append(gid)
                gene_list.append({
                    "rank": rank_i + 1,
                    "gene_name": gid,
                    "probability": round(pred.get("probability") or 0, 4),
                })

        # STEP 1: PubMed batch validation


        online_results = {}
        if HAS_ONLINE_TOOLS:
            clean_symbols = [s for s in gene_symbols
                           if not s.startswith("ENSG") and not s.startswith("Gene_")]
            validate_symbols = clean_symbols[:30]
            print(f"\n   Querying PubMed for {len(validate_symbols)} genes...")
            if validate_symbols:
                try:
                    import time
                    all_results = {"status": "success", "total_validated": 0,
                                   "disease_context": disease_context,
                                   "summary": {"high_evidence_genes": 0,
                                               "medium_evidence_genes": 0,
                                               "low_evidence_genes": 0,
                                               "api_errors": 0},
                                   "per_gene": {}}
                    for batch_start in range(0, len(validate_symbols), 10):
                        batch = validate_symbols[batch_start:batch_start+10]
                        batch_result = batch_validate_genes(batch, disease_context)
                        if batch_result.get("status") == "success":
                            all_results["per_gene"].update(batch_result.get("per_gene", {}))
                            all_results["total_validated"] += batch_result.get("total_validated", 0)
                            for k in ("high_evidence_genes", "medium_evidence_genes", "low_evidence_genes", "api_errors"):
                                all_results["summary"][k] += batch_result.get("summary", {}).get(k, 0)
                        if batch_start + 10 < len(validate_symbols):
                            time.sleep(0.5)  # rate limit

                    online_results = all_results
                    n_high = online_results.get("summary", {}).get("high_evidence_genes", 0)
                    n_med = online_results.get("summary", {}).get("medium_evidence_genes", 0)
                    n_low = online_results.get("summary", {}).get("low_evidence_genes", 0)
                    n_err = online_results.get("summary", {}).get("api_errors", 0)
                    parts = [f"{n_high} high"]
                    if n_med: parts.append(f"{n_med} medium")
                    parts.append(f"{n_low} low")
                    if n_err: parts.append(f"{n_err} API errors")
                    print(f"   PubMed: {', '.join(parts)}")
                    per_gene = online_results.get("per_gene", {})
                    top_pubmed = sorted(
                        [(s, i.get("pubmed", {}).get("pubmed_count", 0)
                          if isinstance(i.get("pubmed"), dict) else 0)
                         for s, i in per_gene.items()],
                        key=lambda x: x[1], reverse=True
                    )[:5]
                    if top_pubmed and top_pubmed[0][1] > 0:
                        print(f"   Top PubMed: {', '.join(f'{g}={c}' for g, c in top_pubmed)}")
                except Exception as e:
                    print(f"   Online query failed: {e}")
        else:
            print("   Online tools not available")

        # STEP 2: structured evidence


        per_gene_summary: dict[str, int] = {}
        if online_results:
            for sym, info in online_results.get('per_gene', {}).items():
                if isinstance(info, dict):
                    pubmed = info.get('pubmed', {})
                    if isinstance(pubmed, dict):
                        per_gene_summary[sym] = pubmed.get('pubmed_count', 0)

        evidence_report = {
            "disease_context": disease_context,
            "total_validated": len(gene_symbols[:30]),
            "summary": online_results.get('summary', {}) if online_results else {},
            "per_gene_pubmed": per_gene_summary,
            "top_genes": gene_list[:15],
        }

        print(f"\n[LiteratureAgent] Online Validation Complete")
        print(f"   PubMed API queries: {len(gene_symbols[:30])} genes")

        return {
            "literature_report": json.dumps(evidence_report, indent=2, ensure_ascii=False),
            "pubmed_evidence": per_gene_summary,
        }

    async def provide_evidence_for_gene(self, gene_symbol: str,
                                         disease_context: str = "cancer") -> dict:
        """单基因证据查询，供辩论协议使用。"""
        import time as _time
        evidence: dict = {"gene_symbol": gene_symbol, "pubmed_count": 0}

        if HAS_ONLINE_TOOLS:
            clean = _clean_symbol(gene_symbol)
            try:
                pub_result = query_pubmed_count(clean, disease_context)
                if pub_result.get("status") == "success":
                    evidence["pubmed_count"] = pub_result.get("pubmed_count", 0)
                _time.sleep(0.3)  # rate limit
            except Exception:
                pass

        return evidence
