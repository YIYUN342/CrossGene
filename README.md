# CrossGene

LLM + GPS 图神经网络的多智能体基因-疾病关联预测系统。

## 核心思路

- **LLM 视角**：多个智能体基于文献、多组学知识预测基因-疾病关联。
- **GPS 网络视角**：图神经网络（GatedGCN + 多头注意力 + FFN）基于基因网络拓扑打分。
- **辩论共识**：MediatorAgent / CriticAgent / MemoryAgent 融合两种视角并迭代优化。

## 运行

```bash
py run.py
```

批量消融实验：

```bash
py batch_run.py
```

## 依赖

- Python 3.13
- TensorFlow（GPS 图神经网络）
- LangChain（LLM 智能体）
- API Key 配置在 `.env`
