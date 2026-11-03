"""
GeneAgent Workflow — iterative LLM-based gene-disease prediction with feedback loop.

Pipeline:
  MetaAgent (plan) → GeneDataAgent (data) →
    ┌─────────────────────────────────────────┐
    │ Iteration loop (max 3 rounds):           │
    │   LLMFabAgent → LiteratureAgent →        │
    │   GeneAnalysisAgent → MemoryAgent →      │
    │   MetaAgent (review → CONTINUE/FINISH)   │
    └─────────────────────────────────────────┘
  → MetaAgent (final conclusion) → END
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from workflow.state import GeneAgentState, create_initial_state
from core.config import Settings
from agents.meta_agent import MetaAgent
from agents.gene_data_agent import GeneDataAgent
from agents.llm_fab_agent import LLMFabAgent
from agents.literature_agent import LiteratureAgent
from agents.gene_analysis_agent import GeneAnalysisAgent
from agents.memory_agent import MemoryAgent


class GeneWorkflow:
    """Iterative gene-disease prediction workflow with feedback loop."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.agents = {}
        self.graph = None
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return
        s = self.settings
        self.agents = {
            "meta": MetaAgent(settings=s),
            "data": GeneDataAgent(settings=s),
            "fab": LLMFabAgent(settings=s),
            "literature": LiteratureAgent(settings=s),
            "analysis": GeneAnalysisAgent(settings=s),
            "memory": MemoryAgent(settings=s),
        }
        print("Connecting GeneAgent agents...")
        for name, agent in self.agents.items():
            await agent._initialize()
        self.graph = self._build_graph()
        self._initialized = True
        print("GeneAgent workflow ready (iterative mode).")

    def _build_graph(self):
        wf = StateGraph(GeneAgentState)

        # Nodes


        wf.add_node("meta_plan", self._meta_plan)
        wf.add_node("data", self.agents["data"].run)
        wf.add_node("fab", self.agents["fab"].run)
        wf.add_node("literature", self.agents["literature"].run)
        wf.add_node("analysis", self.agents["analysis"].run)
        wf.add_node("memory", self.agents["memory"].run)
        wf.add_node("meta_review", self._meta_review)
        wf.add_node("final_conclusion", self._final_conclusion)

        # Edges


        wf.set_entry_point("meta_plan")

        # Conditional: network_enhanced → data; llm_only → fab (skip data)


        wf.add_conditional_edges(
            "meta_plan",
            self._route_after_meta,
            {"data": "data", "fab": "fab"},
        )
        wf.add_edge("data", "fab")
        wf.add_edge("fab", "literature")
        wf.add_edge("literature", "analysis")
        wf.add_edge("analysis", "memory")
        wf.add_edge("memory", "meta_review")

        # Conditional: continue iterating or finish?


        wf.add_conditional_edges(
            "meta_review",
            self._should_continue,
            {
                "continue": "fab",
                "finish": "final_conclusion",
            },
        )
        wf.add_edge("final_conclusion", END)

        return wf.compile()

    # Node wrappers



    async def _meta_plan(self, state: GeneAgentState) -> dict:
        """Initial planning — ensure iteration=0."""
        state["iteration"] = 0
        return await self.agents["meta"].run(state)

    async def _meta_review(self, state: GeneAgentState) -> dict:
        """Review iteration results and decide next step."""
        return await self.agents["meta"].run(state)

    async def _final_conclusion(self, state: GeneAgentState) -> dict:
        """Generate final conclusion after all iterations."""
        return await self.agents["meta"].generate_final_conclusion(state)

    # Routing



    def _route_after_meta(self, state: GeneAgentState) -> str:
        """After MetaAgent: go to GeneDataAgent or skip to LLMFabAgent."""
        mode = state.get("mode", "network_enhanced")
        if mode == "llm_only":
            print("\n   ═══ LLM-Only Mode: skipping network data agent ═══")
            return "fab"
        return "data"

    def _should_continue(self, state: GeneAgentState) -> str:
        """Route to next iteration or to final conclusion."""
        mode = state.get("mode", "network_enhanced")
        # llm_only: no iterations (no ground truth to evaluate improvement)


        if mode == "llm_only":
            return "finish"
        decision = state.get("meta_decision", "")
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", 3)

        if decision.upper() == "FINISH" or iteration > max_iter:
            return "finish"
        return "continue"

    # Run / shutdown



    async def run(self, goal: str, max_iterations: int = 3,
                 forced_dataset: str | None = None) -> dict:
        if not self._initialized:
            await self.initialize()
        state = create_initial_state(goal, max_iterations,
                                     forced_dataset=forced_dataset)
        print(f"\n{'='*60}")
        print(f"GeneAgent Workflow (iterative, max {max_iterations} rounds)")
        print(f"Goal: {goal}")
        if forced_dataset:
            print(f"Dataset (forced): {forced_dataset}")
        print(f"{'='*60}")
        return await self.graph.ainvoke(state)

    async def shutdown(self):
        for agent in self.agents.values():
            await agent._shutdown()
        self._initialized = False
