"""GeneDataAgent — 读取 H5 多组学基因网络数据，为下游智能体提供数据上下文。"""

import json
import re
from typing import Any

from core.base_agent import BaseAgent
from core.config import Settings

try:
    from tools.h5_reader import (
        list_datasets,
        get_dataset_summary,
        query_gene_network,
        select_candidate_genes,
        GENE_DATA_TOOLS,
    )
    list_available_datasets = list_datasets
    HAS_H5_READER = True
except ImportError:
    HAS_H5_READER = False
    GENE_DATA_TOOLS = []
    list_datasets = None
    get_dataset_summary = None
    query_gene_network = None


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else default
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False) if value else default
    return str(value)


def safe_truncate(value: Any, max_len: int, suffix: str = "...", default: str = "N/A") -> str:
    str_value = safe_str(value, default)
    if len(str_value) > max_len:
        return str_value[:max_len] + suffix
    return str_value


SYSTEM_PROMPT = """You are GeneDataAgent - Multi-Omics Data Intelligence Officer.

## 🎯 Core Mission
Inspect and analyze multi-omics gene interaction network datasets to provide data context for gene-disease association prediction.

You work with H5 datasets containing:
- Gene-gene interaction networks (adjacency matrices)
- 64-dimensional multi-omics node features per gene
- Training/validation/test labels for disease association

## 🛠️ Your Toolbox
| Tool | Purpose |
|------|---------|
| `list_available_datasets` | List all datasets with their availability status (H5 file, preprocessed data, trained weights) |
| `get_dataset_summary` | Detailed summary: gene count, network density, degree distribution, label balance, feature stats |
| `query_gene_network` | Inspect a specific gene: its degree, feature stats, and top network neighbors |

## 📌 Workflow
1. **Start with `list_available_datasets`** to see what data is available
2. **Use `get_dataset_summary`** on the most relevant dataset to understand its characteristics
3. **Optionally query specific genes** to understand local network structure
4. **Compile findings** into a structured data context for downstream agents

## 💡 Decision Guidelines
- Which dataset has the best coverage (most genes, good label balance)?
- What is the network density — sparse or dense?
- How imbalanced are the disease labels?
- Which datasets have trained model weights ready for prediction?

## Output Principles
- Report actual data from tool calls (don't invent numbers)
- Note dataset limitations (small size, imbalance, missing weights)
- Recommend which dataset to use for prediction
"""


class GeneDataAgent(BaseAgent):
    """多组学数据智能体：分析 H5 基因网络数据集，输出 data_context。"""

    LOCAL_GENE_DATA_TOOLS = {
        "list_available_datasets",
        "get_dataset_summary",
        "query_gene_network",
        "select_candidate_genes",
    }

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(name="GeneDataAgent", settings=settings)
        self._local_tools = GENE_DATA_TOOLS.copy() if GENE_DATA_TOOLS else []
        self._full_candidates = []
        self._selection_meta = {}

    def _get_system_prompt(self, state: dict[str, Any], default_prompt: str | None = None) -> str:
        return SYSTEM_PROMPT

    async def _get_tools_with_local_tools(self) -> list[dict[str, Any]]:
        return self._local_tools if self._local_tools else []

    async def _handle_local_tool(self, name: str, args: dict[str, Any]) -> str:
        if not HAS_H5_READER:
            return json.dumps({"status": "error", "message": "H5 reader tools not available"})

        if name == "list_available_datasets":
            result = list_available_datasets()
        elif name == "get_dataset_summary":
            result = get_dataset_summary(args.get("dataset_name", ""))
        elif name == "query_gene_network":
            result = query_gene_network(
                args.get("dataset_name", ""),
                args.get("gene_id"),
                args.get("top_k", 5),
            )
        elif name == "select_candidate_genes":
            import os
            if os.environ.get("CROSSGENE_QUICK"):
                result = select_candidate_genes(
                    args.get("dataset_name", "CPDB"),
                    n_test_pos=10, n_test_neg=20, n_hub=10,
                )
            else:
                result = select_candidate_genes(
                    args.get("dataset_name", "CPDB"),
                    n_hub=args.get("n_hub", 20),
                )
            if result.get("status") == "success":
                self._full_candidates = result.get("candidates", [])
                self._selection_meta = {
                    k: v for k, v in result.items() if k != "candidates"
                }
            summary = {
                "status": "success",
                "candidates_selected": result.get("candidates_selected"),
                "selection_method": result.get("selection_method"),
                "test_set_positive": result.get("test_set_positive_available"),
                "test_set_negative": result.get("test_set_negative_available"),
                "note": f"Full gene list ({result.get('candidates_selected')} genes) stored for downstream agents.",
            }
            return json.dumps(summary, indent=2, ensure_ascii=False)
        else:
            result = {"status": "error", "message": f"Unknown tool: {name}"}

        return json.dumps(result, indent=2, ensure_ascii=False)

    def _truncate_tool_output(self, result_str: str, tool_name: str) -> str:
        """Truncate large tool outputs to avoid blowing up the LLM context."""
        max_len = 4000
        if len(result_str) > max_len:
            return result_str[:max_len] + f"\n... [truncated {len(result_str) - max_len} chars from {tool_name} output]"
        return result_str

    def _validate_against_tools(self, response: str, tool_results: list[dict[str, Any]]) -> str:
        """后校验：标记无法溯源到工具输出的数值声明。"""
        if not tool_results or not response:
            return response

        tool_numbers: set[str] = set()
        for tr in tool_results:
            result_str = tr.get("result", "")
            if not result_str or result_str.startswith("[ERROR]"):
                continue
            import re
            for m in re.finditer(r'\b(\d+(?:\.\d+)?)\b', str(result_str)):
                tool_numbers.add(m.group(1))

        suspicious = []
        for m in re.finditer(r'\b(\d+(?:\.\d+)?)\b', response):
            num = m.group(1)
            if num in ('0', '1', '2', '3') or len(num) == 4:
                continue
            if num not in tool_numbers:
                suspicious.append(num)

        if suspicious:
            unique = list(dict.fromkeys(suspicious))
            flagged = unique[:10]
            note = (
                f"<!-- ⚠️ HALLUCINATION CHECK: {len(suspicious)} numeric claims not "
                f"found in tool outputs. Unverified: {flagged}. "
                f"Trust ONLY numbers from tool JSON above. -->\n\n"
            )
            return note + response

        return response

    async def autonomous_thinking(
        self, prompt: str, state: dict[str, Any],
        system_message: str | None = None, max_iterations: int = 10,
        tool_call_required: bool = False,
    ) -> dict[str, Any]:
        """覆盖基类，处理本地 H5 工具并防幻觉。"""
        if not self.llm:
            return {"response": "[ERROR] LLM not configured", "tool_calls": [],
                    "tool_results": [], "iterations": 0}

        tools = await self._get_tools_with_local_tools()
        messages: list[dict[str, Any]] = []
        final_sp = self._get_system_prompt(state, system_message)
        if final_sp:
            messages.append({"role": "system", "content": final_sp})

        context_str = ""
        if state:
            context_str = f"\n\nCurrent context:\n{state}"
        messages.append({"role": "user", "content": prompt + context_str})

        all_tool_calls, all_tool_results = [], []
        iterations, response = 0, None
        _last_tool_name, _consecutive_count = None, 0
        _tool_retry_used = False

        while iterations < max_iterations:
            iterations += 1
            response = await self.llm.ainvoke(
                messages, tools=tools if tools else None)
            if not self.llm.has_tool_calls(response):
                if (tool_call_required and tools and not all_tool_calls
                        and not _tool_retry_used):
                    _tool_retry_used = True
                    print(f"\n[GeneDataAgent] No tool calls — retrying with tool-use directive...")
                    messages.append({
                        "role": "user",
                        "content": (
                            "⚠️ CRITICAL: You MUST call the available tools before answering. "
                            "You have NO access to file contents, database records, or real data "
                            "without calling tools. If you answer without calling tools, your "
                            "response will be DISCARDED. Call a tool NOW."
                        ),
                    })
                    continue
                break

            tool_calls = self.llm.get_tool_calls(response)
            messages.append({
                "role": "assistant",
                "content": response.content if hasattr(response, 'content') and response.content else "",
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                tool_name, tool_args, tool_id = tc["name"], tc["args"], tc["id"]

                tool_type = "Local" if tool_name in self.LOCAL_GENE_DATA_TOOLS else "MCP"
                if tool_name == _last_tool_name:
                    _consecutive_count += 1
                    print(f"\r   [GeneDataAgent] {tool_name} ×{_consecutive_count}", end="", flush=True)
                else:
                    if _last_tool_name is not None and _consecutive_count > 1:
                        print()
                    _consecutive_count, _last_tool_name = 1, tool_name
                    print(f"\n[GeneDataAgent] {tool_type}: {tool_name}")
                    print(f"   Args: {str(tool_args)[:200]}")

                all_tool_calls.append(tc)

                try:
                    if tool_name in self.LOCAL_GENE_DATA_TOOLS:
                        result_str = await self._handle_local_tool(tool_name, tool_args)
                    else:
                        result_str = f"[ERROR] Unknown tool: {tool_name}"
                    result_str = self._truncate_tool_output(result_str, tool_name)
                except Exception as e:
                    result_str = f"[ERROR] {e}"

                all_tool_results.append({"tool": tool_name, "result": result_str})
                messages.append(self.llm.create_tool_message(tool_id, result_str))

        final_response = response.content if response and hasattr(response, "content") else ""
        return {"response": final_response, "tool_calls": all_tool_calls,
                "tool_results": all_tool_results, "iterations": iterations,
                "tool_retry_used": _tool_retry_used}

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Read multi-omics gene network data — direct function calls."""
        print(f"\n{'='*60}")
        print(f"[GeneDataAgent] Multi-Omics Data Analysis (direct)")
        print(f"{'='*60}")

        goal = state.get("goal", "")
        plan = state.get("plan", {}) or {}
        gene_categories = plan.get("gene_categories", [])
        disease_context = plan.get("disease_context", "")

        if not HAS_H5_READER:
            return {
                "data_context": json.dumps({"status": "error", "message": "H5 reader not available"}),
                "candidate_genes": [],
            }

        dataset_name = plan.get("dataset", "CPDB") if plan else "CPDB"
        if not dataset_name or not isinstance(dataset_name, str):
            dataset_name = "CPDB"
        print(f"\n   MetaAgent selected dataset: {dataset_name}")

        print(f"\n   Checking datasets...")
        datasets = list_available_datasets()
        ready = datasets.get("ready", 0)
        print(f"   {ready}/{datasets.get('total', 0)} datasets ready")

        print(f"\n   Selecting candidate genes from {dataset_name}...")
        import os as _os
        if _os.environ.get("CROSSGENE_QUICK"):
            selection = select_candidate_genes(dataset_name, n_test_pos=10, n_test_neg=20, n_hub=10)
        else:
            selection = select_candidate_genes(dataset_name, n_hub=20)

        if selection.get("status") != "success":
            return {"data_context": json.dumps({"status": "error", "message": f"Failed to read {dataset_name}"})}

        self._full_candidates = selection.get("candidates", [])
        self._selection_meta = {k: v for k, v in selection.items() if k != "candidates"}

        n_genes = selection.get("total_genes_in_network", "?")
        n_edges = selection.get("total_edges", "?")
        n_candidates = selection.get("candidates_selected", 0)
        print(f"   Network: {n_genes} genes, {n_edges} edges")
        print(f"   Selected: {n_candidates} candidates ({selection.get('n_test_genes', '?')} test + "
              f"{selection.get('n_hub_genes', '?')} hub)")

        category_hint = ""
        if gene_categories:
            category_hint = f"Focus on {', '.join(gene_categories)} genes. "
        synthesis_text = (
            f"{dataset_name} gene interaction network: {n_genes} genes, {n_edges} edges. "
            f"{category_hint}"
            f"{n_candidates} genes selected for disease association prediction "
            f"({selection.get('n_test_genes', '?')} with ground truth labels)."
        )

        tool_results = [
            {"tool": "list_available_datasets", "result": json.dumps(datasets)},
            {"tool": "select_candidate_genes", "result": json.dumps(selection)},
        ]

        print(f"\n[GeneDataAgent] Done — {n_candidates} genes selected")

        data_context_str, candidate_genes_list = self._build_data_context(
            synthesis_text, tool_results, dataset_name)

        return {
            "data_context": data_context_str,
            "candidate_genes": candidate_genes_list,
        }

    def _extract_my_task(self, plan: str | dict, agent_name: str) -> str:
        """从 plan 中提取本智能体的任务。"""
        if not plan:
            return "Analyze available multi-omics gene network datasets"
        if isinstance(plan, dict):
            agent_tasks = plan.get("agent_tasks", {})
            for key in (agent_name, "DataAgent"):
                task = agent_tasks.get(key, "")
                if task and str(task).upper() != "SKIP":
                    return task
        return "Analyze gene network data and recommend datasets for prediction"

    def _build_data_context(self, response: str, tool_results: list[dict[str, Any]],
                            dataset_name: str = "CPDB") -> str:
        """构建结构化的 JSON 数据上下文。"""
        datasets_analyzed = []
        candidate_genes = getattr(self, '_full_candidates', [])
        selection_meta = getattr(self, '_selection_meta', {})

        for tr in tool_results:
            tool_name = tr.get("tool", "")
            result_str = tr.get("result", "")
            if result_str.startswith("[ERROR]"):
                continue

            try:
                data = json.loads(result_str) if isinstance(result_str, str) else result_str
            except (json.JSONDecodeError, TypeError):
                continue

            if data.get("status") != "success":
                continue

            if tool_name == "select_candidate_genes":
                datasets_analyzed.append({
                    "dataset_name": data.get("dataset_name"),
                    "n_genes": data.get("total_genes_in_network"),
                    "n_edges": data.get("total_edges"),
                    "network_density": round(data.get("total_edges", 0) / (data.get("total_genes_in_network", 1) ** 2), 6),
                    "mean_degree": data.get("network_mean_degree"),
                })

        synthesis = safe_truncate(response, 1000, default="Data analysis completed")

        gps_scores = self._load_gps_scores(dataset_name)
        if gps_scores:
            for g in candidate_genes:
                name = g.get("gene_name", "")
                if name in gps_scores:
                    g["gps_score"] = round(gps_scores[name], 4)
            print(f"   Attached GPS scores to {sum(1 for g in candidate_genes if 'gps_score' in g)}/{len(candidate_genes)} genes")

        context = {
            "dataset_name": dataset_name,
            "datasets_analyzed": datasets_analyzed,
            "selection_meta": selection_meta,
            "synthesis": synthesis,
            "tool_calls": len(tool_results),
        }
        return json.dumps(context, indent=2, ensure_ascii=False), candidate_genes

    def _load_gps_scores(self, dataset_name: str = "CPDB") -> dict:
        """加载预计算的 GPS 预测分数。"""
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        specific_path = os.path.join(base_dir, f"gps_predictions_{dataset_name}.json")
        if os.path.exists(specific_path):
            try:
                with open(specific_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return self._parse_gps_scores(data)
            except Exception:
                pass

        gps_path = os.path.join(base_dir, "gps_predictions.json")
        if not os.path.exists(gps_path):
            return {}
        try:
            with open(gps_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self._parse_gps_scores(data)
        except Exception:
            return {}

    @staticmethod
    def _parse_gps_scores(data: dict) -> dict:
        """解析 GPS 预测分数。"""
        predictions = data.get("predictions", {})
        result = {}
        for k, v in predictions.items():
            score = v.get("gps_score")
            if score is not None:
                result[k] = score
        return result
