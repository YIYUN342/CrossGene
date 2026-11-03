"""Shared state for GeneAgent workflow with iterative feedback loop."""
from typing import TypedDict, Optional


class GeneAgentState(TypedDict, total=False):
    # Input


    goal: str

    # MetaAgent output


    plan: Optional[dict]
    disease_context: str
    mode: str                     # "network_enhanced" | "llm_only"          # e.g. "breast cancer" — flows to LiteratureAgent

    # GeneDataAgent output


    data_context: Optional[str]   # lightweight: datasets + selection_meta + synthesis (no gene list)
    candidate_genes: Optional[list]  # native list, avoids JSON round-trip

    # LLMFabAgent output


    fab_results: Optional[dict]

    # LiteratureAgent output


    literature_report: Optional[str]
    pubmed_evidence: Optional[dict]  # structured: gene_name → pubmed_count

    # GeneAnalysisAgent output


    analysis_report: Optional[str]

    # MemoryAgent output


    memory_insights: Optional[str]

    # Iteration control


    iteration: int
    max_iterations: int
    iteration_history: list
    all_round_results: list         # per-round fab_results summaries, never overwritten
    meta_decision: str
    feedback: str

    # Final


    final_conclusion: Optional[str]

    # Override


    forced_dataset: Optional[str]


def create_initial_state(goal: str, max_iterations: int = 3,
                         forced_dataset: str | None = None) -> GeneAgentState:
    return {
        "goal": goal,
        "plan": None,
        "disease_context": "",
        "mode": "",
        "data_context": None,
        "candidate_genes": None,
        "fab_results": None,
        "literature_report": None,
        "pubmed_evidence": None,
        "analysis_report": None,
        "memory_insights": None,
        "iteration": 0,
        "max_iterations": max_iterations,
        "iteration_history": [],
        "all_round_results": [],
        "meta_decision": "",
        "feedback": "",
        "final_conclusion": None,
        "forced_dataset": forced_dataset,
    }
