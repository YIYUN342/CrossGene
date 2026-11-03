#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Online database query tools for LiteratureAgent.

Queries free public APIs to validate gene-disease associations:
  - MyGene.info: Gene symbol → Ensembl/Entrez ID mapping + summary
  - NCBI E-utilities: PubMed publication counts, OMIM entries
  - Open Targets Platform: Gene-disease association scores

All APIs are free, no API key required for basic queries.
"""

import json
import urllib.request
import urllib.parse
from typing import Dict, Any


def query_mygene(gene_symbol: str) -> Dict[str, Any]:
    """
    Query MyGene.info for gene information.

    Free REST API. Returns Ensembl ID, Entrez ID, gene summary, genomic location.
    Rate limit: ~3 requests/second without API key.

    Example: query_mygene("TP53") → {entrez_id, ensembl_id, summary, ...}
    """
    try:
        url = f"https://mygene.info/v3/query?q=symbol:{urllib.parse.quote(gene_symbol)}&fields=entrezgene,ensembl.gene,summary,name,symbol"
        req = urllib.request.Request(url, headers={"User-Agent": "GenePrediction/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        if data.get("hits") and len(data["hits"]) > 0:
            hit = data["hits"][0]
            return {
                "status": "success",
                "symbol": hit.get("symbol", gene_symbol),
                "name": hit.get("name", ""),
                "entrez_id": hit.get("entrezgene", ""),
                "ensembl_id": hit.get("ensembl", {}).get("gene", ""),
                "summary": hit.get("summary", "")[:500] if hit.get("summary") else "",
            }
        return {"status": "not_found", "symbol": gene_symbol}
    except Exception as e:
        return {"status": "error", "symbol": gene_symbol, "message": str(e)}


def query_ncbi_gene(gene_symbol: str) -> Dict[str, Any]:
    """
    Query NCBI Gene database via E-utilities for disease associations.

    Free, no API key needed for <3 requests/second.
    Returns: OMIM IDs, disease descriptions, PubMed count.
    """
    try:
        # Step 1: Search for gene ID


        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = urllib.parse.urlencode({
            "db": "gene",
            "term": f"{gene_symbol}[sym] AND human[orgn]",
            "retmax": 1,
            "retmode": "json",
        })
        req = urllib.request.Request(f"{search_url}?{params}",
                                     headers={"User-Agent": "GenePrediction/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            search_data = json.loads(resp.read().decode())

        gene_ids = search_data.get("esearchresult", {}).get("idlist", [])
        if not gene_ids:
            return {"status": "not_found", "symbol": gene_symbol}

        gene_id = gene_ids[0]

        # Step 2: Fetch gene summary (includes phenotype/disease info)


        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = urllib.parse.urlencode({
            "db": "gene",
            "id": gene_id,
            "retmode": "json",
        })
        req = urllib.request.Request(f"{summary_url}?{params}",
                                     headers={"User-Agent": "GenePrediction/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            summary_data = json.loads(resp.read().decode())

        gene_info = summary_data.get("result", {}).get(gene_id, {})
        description = gene_info.get("description", "")
        summary_text = gene_info.get("summary", "")[:1000] if gene_info.get("summary") else ""

        # Extract phenotypes/disease from summary


        phenotypes = []
        if "phenotype" in str(gene_info).lower():
            phenotypes.append("Has phenotype associations")

        return {
            "status": "success",
            "symbol": gene_symbol,
            "ncbi_gene_id": gene_id,
            "description": description,
            "summary": summary_text,
            "nomenclature": gene_info.get("nomenclaturename", ""),
            "has_phenotype_associations": len(phenotypes) > 0,
        }
    except Exception as e:
        return {"status": "error", "symbol": gene_symbol, "message": str(e)}


def query_pubmed_count(gene_symbol: str, disease: str = "cancer") -> Dict[str, Any]:
    """
    Query PubMed for publication count linking a gene to a disease.

    Returns the number of publications matching "gene_symbol AND disease".
    Uses [All Fields] tag for PubMed compatibility (NOT [sym] which is
    specific to NCBI Gene database).
    """
    try:
        # Use [All Fields] for both gene name and disease — [sym] is NOT a


        # valid PubMed search field (it belongs to the NCBI Gene database).


        disease_term = f'"{disease}"' if " " in disease else disease
        query = f'{gene_symbol}[All Fields] AND {disease_term}[All Fields]'
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = urllib.parse.urlencode({
            "db": "pubmed",
            "term": query,
            "retmax": 0,
            "retmode": "json",
        })
        req = urllib.request.Request(f"{search_url}?{params}",
                                     headers={"User-Agent": "GenePrediction/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        count = int(data.get("esearchresult", {}).get("count", 0))
        return {
            "status": "success",
            "gene": gene_symbol,
            "disease": disease,
            "pubmed_count": count,
            "query": f"{gene_symbol} AND {disease}",
        }
    except Exception as e:
        return {"status": "error", "gene": gene_symbol, "message": str(e)}


def query_open_targets(gene_symbol: str) -> Dict[str, Any]:
    """
    Query Open Targets Platform for gene-disease association scores.

    Free REST API. Returns overall association score and top associated diseases.
    """
    try:
        # First get Ensembl ID from gene symbol


        search_url = "https://api.platform.opentargets.org/api/v4/graphql"
        query = """
        query SearchGene($symbol: String!) {
          search(query: $symbol, entityNames: ["target"]) {
            hits {
              id
              name
            }
          }
        }
        """
        payload = json.dumps({"query": query, "variables": {"symbol": gene_symbol}}).encode()
        req = urllib.request.Request(search_url, data=payload,
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "GenePrediction/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        hits = data.get("data", {}).get("search", {}).get("hits", [])
        if not hits:
            return {"status": "not_found", "symbol": gene_symbol}

        ensembl_id = hits[0]["id"]

        # Step 2: Get associated diseases


        disease_query = """
        query AssociatedDiseases($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            associatedDiseases(page: {size: 5}) {
              rows {
                disease {
                  name
                }
                score
              }
            }
          }
        }
        """
        payload = json.dumps({"query": disease_query,
                              "variables": {"ensemblId": ensembl_id}}).encode()
        req = urllib.request.Request(search_url, data=payload,
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "GenePrediction/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        rows = data.get("data", {}).get("target", {}).get("associatedDiseases", {}).get("rows", [])
        diseases = [{"disease": r["disease"]["name"], "score": r["score"]} for r in rows]

        return {
            "status": "success",
            "symbol": gene_symbol,
            "ensembl_id": ensembl_id,
            "top_diseases": diseases,
        }
    except Exception as e:
        return {"status": "error", "symbol": gene_symbol, "message": str(e)}


def _clean_symbol(s: str) -> str:
    """Remove bytes-prefix artifacts from gene symbols."""
    s = str(s)
    for prefix in ["b'", 'b"', "'", '"']:
        if s.startswith(prefix):
            s = s[len(prefix):]
        if s.endswith(prefix):
            s = s[:-len(prefix)]
    return s.strip()


def batch_validate_genes(gene_symbols: list, disease_context: str = "cancer") -> Dict[str, Any]:
    """
    Batch validate a list of gene symbols against multiple online databases.

    For each gene, queries PubMed and MyGene.info.
    Returns a summary of validation results.
    """
    results = {}
    for i, symbol in enumerate(gene_symbols):
        symbol = _clean_symbol(symbol)
        # Skip non-gene-symbol strings (ENSG IDs, "Gene_X")


        if symbol.startswith("ENSG") or symbol.startswith("Gene_"):
            results[symbol] = {"symbol": symbol, "skipped": True, "reason": "Not a gene symbol"}
            continue
        print(f"  [{i+1}/{len(gene_symbols)}] Querying {symbol}...")
        result = {
            "symbol": symbol,
            "pubmed": query_pubmed_count(symbol, disease_context),
        }
        results[symbol] = result

    # Summary — separate API errors from genuine low counts


    n_high = 0; n_medium = 0; n_low = 0; n_errors = 0
    for r in results.values():
        pubmed = r.get("pubmed", {})
        if not isinstance(pubmed, dict):
            n_errors += 1
            continue
        if pubmed.get("status") == "error":
            n_errors += 1
            continue
        count = pubmed.get("pubmed_count", 0)
        if count > 100:
            n_high += 1
        elif 10 <= count <= 100:
            n_medium += 1
        else:
            n_low += 1

    return {
        "status": "success",
        "total_validated": len(results),
        "disease_context": disease_context,
        "summary": {
            "high_evidence_genes": n_high,
            "medium_evidence_genes": n_medium,
            "low_evidence_genes": n_low,
            "api_errors": n_errors,
        },
        "per_gene": results,
    }


# Tool definitions for LLM agents


ONLINE_DB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_mygene",
            "description": """Query MyGene.info for gene information (Ensembl ID, Entrez ID, gene summary).
Uses the free MyGene.info REST API. No API key required.
Use this to verify gene symbols and get official gene names.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "gene_symbol": {
                        "type": "string",
                        "description": "Gene symbol (e.g., 'TP53', 'BRCA1', 'KRAS')"
                    }
                },
                "required": ["gene_symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_pubmed_count",
            "description": """Count PubMed publications linking a gene to a disease.
Queries NCBI PubMed via E-utilities (free, no API key).
Returns the number of papers matching 'gene_symbol AND disease'.
High count (>100) = well-studied; Medium (10-100) = some evidence; Low (<10) = limited evidence.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "gene_symbol": {
                        "type": "string",
                        "description": "Gene symbol (e.g., 'TP53')"
                    },
                    "disease": {
                        "type": "string",
                        "description": "Disease context (default: 'cancer')"
                    }
                },
                "required": ["gene_symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_open_targets",
            "description": """Query Open Targets Platform for gene-disease association scores.
Free GraphQL API. Returns top 5 associated diseases with association scores.
Use this to find which diseases a gene is most strongly associated with.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "gene_symbol": {
                        "type": "string",
                        "description": "Gene symbol (e.g., 'TP53')"
                    }
                },
                "required": ["gene_symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "batch_validate_genes",
            "description": """Batch validate multiple gene symbols against PubMed.
Queries each gene against a disease context and returns publication counts.
Use this for efficient batch validation of top predicted genes.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "gene_symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of gene symbols to validate"
                    },
                    "disease_context": {
                        "type": "string",
                        "description": "Disease context for PubMed query (default: 'cancer')"
                    }
                },
                "required": ["gene_symbols"]
            }
        }
    },
]


if __name__ == '__main__':
    # Quick test


    print("Testing MyGene.info...")
    print(json.dumps(query_mygene("TP53"), indent=2))
    print("\nTesting PubMed count...")
    print(json.dumps(query_pubmed_count("TP53", "cancer"), indent=2))
    print("\nTesting Open Targets...")
    print(json.dumps(query_open_targets("TP53"), indent=2))
