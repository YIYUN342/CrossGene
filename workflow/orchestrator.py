"""GeneAgentOrchestrator — 基于 MessageBus 的事件驱动多智能体辩论编排。"""
from __future__ import annotations

import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Settings
from core.messaging import MessageBus
from core.debate import extract_disputed_genes
from workflow.state import GeneAgentState, create_initial_state


class GeneAgentOrchestrator:
    """Event-driven multi-agent orchestrator with debate protocol.

    Ablation modes (set via `ablation` parameter or CROSSGENE_ABLATION env var):
      - None / "full":  all components active
      - "no_critic":    skip CriticAgent challenges
      - "no_mediator":  skip debate entirely (weighted ensemble only)
      - "no_memory":    skip MemoryAgent insights
    """

    def __init__(self, settings: Settings | None = None,
                 ablation: str | None = None):
        self.settings = settings or Settings()
        self.bus = MessageBus()
        self.agents: dict[str, object] = {}
        self._initialized = False
        self.ablation = ablation or os.environ.get("CROSSGENE_ABLATION")

    async def initialize(self):
        if self._initialized:
            return
        s = self.settings

        # Import agents (lazy to avoid circular imports)


        from agents.meta_agent import MetaAgent as _Planner
        from agents.gene_data_agent import GeneDataAgent as _Data
        from agents.llm_fab_agent import LLMFabAgent as _Knowledge
        from agents.literature_agent import LiteratureAgent as _Lit
        from agents.gene_analysis_agent import GeneAnalysisAgent as _Analysis
        from agents.memory_agent import MemoryAgent as _Memory
        from agents.critic_agent import CriticAgent as _Critic
        from agents.mediator_agent import MediatorAgent as _Mediator

        self.agents = {
            "planner": _Planner(settings=s),
            "data": _Data(settings=s),
            "knowledge": _Knowledge(settings=s),
            "literature": _Lit(settings=s),
            "analysis": _Analysis(settings=s),
            "memory": _Memory(settings=s),
            "critic": _Critic(settings=s),
            "mediator": _Mediator(settings=s),
        }

        for name, agent in self.agents.items():
            agent.attach_bus(self.bus)  # type: ignore[union-attr]
            await self.bus.register(name)
            await agent._initialize()  # type: ignore[union-attr]

        self._initialized = True
        print("GeneAgent multi-agent system ready (debate protocol).")

    # Main entry point



    async def run(self, goal: str, max_iterations: int = 3) -> dict:
        print("=" * 70)
        print("GeneAgent — Debate-Driven Multi-Agent Gene-Disease Prediction")
        print(f"Goal: {goal}")
        print(f"Max iterations: {max_iterations}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        if not self._initialized:
            await self.initialize()

        state = create_initial_state(goal, max_iterations)

        try:
            # Phase 1: Planning


            state = await self._phase_planning(state)

            # Phase 2: Data loading (network_enhanced only)


            if state.get("mode") == "network_enhanced":
                state = await self._phase_data(state)

            # Phase 3: Initial prediction


            state = await self._phase_predict(state)

            # Phase 4: Validation + Analysis (parallel)


            state = await self._phase_validate_analyze(state)

            # Phase 5: Debate (network_enhanced only) + memory + conclusion


            state = await self._phase_debate_conclude(state)

            return dict(state)

        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
            return dict(state)
        finally:
            await self.shutdown()

    # Phase implementations



    async def _phase_planning(self, state: GeneAgentState) -> GeneAgentState:
        """Phase 1: Orchestrator (MetaAgent) plans the experiment."""
        planner = self.agents["planner"]
        state["iteration"] = 0
        result = await planner.run(dict(state))  # type: ignore[union-attr]
        state.update(result)
        return state

    async def _phase_data(self, state: GeneAgentState) -> GeneAgentState:
        """Phase 2: NetworkDataAgent loads H5 network data."""
        data_agent = self.agents["data"]
        result = await data_agent.run(dict(state))  # type: ignore[union-attr]
        state.update(result)
        return state

    async def _phase_predict(self, state: GeneAgentState) -> GeneAgentState:
        """Phase 3: KnowledgeAgent predicts gene-disease associations."""
        knowledge = self.agents["knowledge"]
        state["iteration"] = state.get("iteration", 0) or 1
        result = await knowledge.run(dict(state))  # type: ignore[union-attr]
        state.update(result)
        return state

    async def _phase_validate_analyze(self, state: GeneAgentState) -> GeneAgentState:
        """Phase 4: LiteratureAgent → AnalysisAgent (sequential —
        AnalysisAgent needs LiteratureAgent's pubmed_evidence output)."""
        lit = self.agents["literature"]
        analysis = self.agents["analysis"]

        # Literature first (I/O-bound PubMed API queries)


        lit_result = await lit.run(dict(state))  # type: ignore[union-attr]
        state.update(lit_result)

        # Analysis second (needs pubmed_evidence from Literature)


        analysis_result = await analysis.run(dict(state))  # type: ignore[union-attr]
        state.update(analysis_result)

        return state

    async def _phase_debate_conclude(self, state: GeneAgentState) -> GeneAgentState:
        """Phase 5: Debate (if needed) + Memory + Final Conclusion."""
        mode = state.get("mode", "network_enhanced")

        if mode == "network_enhanced":
            state = await self._run_debates(state)

        # ABLATION: no_memory → skip MemoryAgent


        if self.ablation != "no_memory":
            memory = self.agents["memory"]
            mem_result = await memory.run(dict(state))  # type: ignore[union-attr]
            state.update(mem_result)
        else:
            # minimal fallback memory_insights


            state["memory_insights"] = '{"note": "ablation: no_memory"}'

        # MetaAgent review + final conclusion


        planner = self.agents["planner"]
        review_result = await planner.run(dict(state))  # type: ignore[union-attr]
        state.update(review_result)

        conclusion_result = await planner.generate_final_conclusion(dict(state))  # type: ignore[union-attr]
        state.update(conclusion_result)

        return state

    # Debate orchestration



    async def _run_debates(self, state: GeneAgentState) -> GeneAgentState:
        """Identify disputed genes and run per-gene debates in parallel."""
        fab = state.get("fab_results", {})
        predictions = fab.get("predictions", {}) if isinstance(fab, dict) else {}
        pubmed_evidence = state.get("pubmed_evidence", {})

        disputed = extract_disputed_genes(predictions, pubmed_evidence, threshold=0.3)
        if not disputed:
            print("\n    No disputed genes — all predictions aligned.")
            state["debate_resolutions"] = {}
            state["stalemate_genes"] = []
            return state

        print(f"\n{'='*60}")
        print(f"⚔️  Debate Phase: {len(disputed)} disputed genes")
        print(f"{'='*60}")

        # Run debates in parallel (each gene's debate is independent)


        mediator = self.agents["mediator"]
        literature = self.agents["literature"]
        critic = self.agents["critic"]

        debate_tasks = []
        for gene in disputed[:30]:  # cap at 30 to control costs
            task = asyncio.create_task(
                self._debate_one_gene(
                    gene, predictions, pubmed_evidence,
                    mediator, literature, critic, state,
                )
            )
            debate_tasks.append(task)

        results = await asyncio.gather(*debate_tasks, return_exceptions=True)

        resolutions: dict[str, dict] = {}
        stalemates: list[str] = []
        for gene, result in zip(disputed[:30], results):
            if isinstance(result, Exception):
                print(f"   ️ Debate for {gene} failed: {result}")
                continue
            if result:
                resolutions[gene] = result.to_dict()
                if result.status == "stalemate":
                    stalemates.append(gene)

        n_resolved = len(resolutions) - len(stalemates)
        print(f"\n   ⚔️  Debate complete: {n_resolved} resolved, "
              f"{len(stalemates)} stalemated")

        state["debate_resolutions"] = resolutions
        state["stalemate_genes"] = stalemates
        return state

    async def _debate_one_gene(
        self, gene_symbol: str,
        predictions: dict, pubmed_evidence: dict,
        mediator, literature, critic,
        state: GeneAgentState,
    ):
        """Run a debate for one gene, respecting ablation mode."""
        pred = predictions.get(gene_symbol, {})
        if not isinstance(pred, dict):
            return None

        disease_context = state.get("disease_context", "cancer")

        # Round 1: Positions (always constructed)


        knowledge_pos = {
            "probability": pred.get("llm_prob") or pred.get("probability"),
            "confidence": pred.get("confidence", "?"),
            "evidence_type": pred.get("evidence_type", "?"),
            "rationale": pred.get("rationale", "No rationale provided"),
        }
        network_pos = {
            "gps_score": pred.get("gps_score"),
            "gene_degree": pred.get("gene_degree"),
            "degree_tier": pred.get("degree_tier"),
            "top_neighbors": pred.get("top_neighbors", []),
        }

        llm_prob = knowledge_pos["probability"] or 0.5
        gps_score = network_pos["gps_score"] or 0.5

        # ABLATION: no_mediator → skip debate, use weighted ensemble


        if self.ablation == "no_mediator":
            pubmed = pubmed_evidence.get(gene_symbol, 0)
            if pubmed > 100:
                llm_w, gps_w = 0.7, 0.3
            elif pubmed > 10:
                llm_w, gps_w = 0.55, 0.45
            else:
                llm_w, gps_w = 0.5, 0.5
            final = round(llm_w * llm_prob + gps_w * gps_score, 4)
            from agents.consensus import ConsensusRecord
            return ConsensusRecord(
                gene_symbol=gene_symbol, status="resolved",
                final_probability=final, llm_weight=llm_w, gps_weight=gps_w,
                key_evidence=[f"PubMed: {pubmed}", f"LLM: {llm_prob}", f"GPS: {gps_score}"],
                remaining_uncertainty="No debate (ablation: no_mediator)",
                debate_rounds=0,
            )

        # Round 2: Evidence (always collected)


        evidence = await literature.provide_evidence_for_gene(gene_symbol, disease_context)

        # ABLATION: no_critic → empty challenges


        challenges: list[dict] = []
        if self.ablation != "no_critic":
            challenges_data = [
                {"gene_name": gene_symbol, "probability": llm_prob,
                 "confidence": knowledge_pos["confidence"],
                 "evidence_type": knowledge_pos["evidence_type"],
                 "gps_score": gps_score},
            ]
            critic_result = await critic.generate_challenges(challenges_data, pubmed_evidence)
            challenges = [c for c in critic_result if c.get("gene") == gene_symbol]

        gene_challenges = challenges

        pubmed_count = evidence.get("pubmed_count", "?")
        print(f"   ⚔️  {gene_symbol}: LLM={llm_prob:.4f}, "
              f"GPS={gps_score:.4f}, PubMed={pubmed_count}"
              + (f" [{len(gene_challenges)} challenges]" if gene_challenges else ""))

        # Rounds 3-4: Mediation (skip if no_mediator — already handled above)


        result = await mediator.moderate_debate(
            gene_symbol=gene_symbol,
            knowledge_position=knowledge_pos,
            network_position=network_pos,
            evidence=evidence,
            challenges=gene_challenges,
        )

        return result


    async def shutdown(self):
        for agent in self.agents.values():
            await agent._shutdown()  # type: ignore[union-attr]
        self._initialized = False
