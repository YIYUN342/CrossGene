# 论文摘要

## 基于大语言模型多智能体协作与实时数据交叉验证的基因-疾病关联预测框架

### 摘要

**研究背景**：基因-疾病关联预测是理解疾病分子机制和发现药物靶点的核心任务。传统方法依赖图神经网络（如GPS Graphormer）在基因互作网络上进行监督学习，但面临训练数据稀缺、模型泛化能力有限等挑战。大语言模型（LLM）在海量生物医学文献上预训练，蕴含丰富的基因-疾病关联知识，然而其固有的幻觉问题严重制约了在生物医学等高可靠性场景中的应用。

**方法**：本文提出GeneAgent，一个基于LLM的多智能体协作框架，将LLM训练知识转化为可验证的基因-疾病关联预测。该框架包含五个专业化智能体：① MetaAgent负责实验规划与假设生成；② GeneDataAgent通过工具调用读取H5多组学基因网络数据（13,627个基因，504,378条边）；③ LLMFabAgent基于LLM训练知识直接预测基因的疾病关联概率与证据类型；④ GeneAnalysisAgent结合COSMIC癌症基因数据库进行生物学机理解读；⑤ LiteratureAgent调用PubMed、MyGene、Open Targets等公共API进行实时文献验证。为抑制LLM幻觉，我们设计了三层防护机制：强制工具调用（tool_call_required）、工具返回数据锚定校验（grounded output validation）、以及LLM知识与真实API数据的交叉验证（cross-validation）。

**结果**：在CPDB人类基因互作网络上验证了该框架的有效性。LLMFabAgent基于训练知识预测了15个基因的疾病关联概率（TP53: 0.99, KRAS: 0.99, BRCA1: 0.99, EGFR: 0.98等），其中10个高置信度基因（67%）与COSMIC癌症基因普查数据库重合。通过PubMed API实时交叉验证，全部10个高置信度预测均被真实文献数据确认（EGFR: 64,359篇论文, MYC: 38,816篇, TP53: 33,950篇, KRAS: 30,259篇），而TTN、AHNAK等5个低置信度基因被正确识别为缺乏文献证据支持。GeneDataAgent成功从H5文件中加载了3个数据集（CPDB、STRINGdb、PCNET）的真实网络统计信息，消除了数据幻觉。

**结论**：GeneAgent证明了将LLM作为生物医学知识库的可行性——当与严格的工具调用约束和实时API交叉验证相结合时，LLM能够产生与已知生物学知识高度一致且可验证的基因-疾病关联预测。该框架为LLM在生物医学高可靠性场景中的应用提供了一个可复现的范式，其"LLM知识预测-实时数据验证"的双引擎架构可推广至药物重定位、变异致病性解读等相关任务。

**关键词**：大语言模型；多智能体系统；基因-疾病关联预测；幻觉抑制；交叉验证；PubMed API；生物医学知识库

---

### Abstract (English)

**Background**: Gene-disease association prediction is fundamental to understanding disease mechanisms and identifying therapeutic targets. Traditional approaches rely on graph neural networks trained on gene interaction networks, which face challenges including limited training data and constrained generalization. Large language models (LLMs), pre-trained on vast biomedical corpora, embed rich gene-disease association knowledge, yet their inherent hallucination problem severely limits their application in high-stakes biomedical settings.

**Methods**: We propose GeneAgent, a multi-agent LLM framework that transforms LLM training knowledge into verifiable gene-disease association predictions. The framework comprises five specialized agents: ① MetaAgent for experimental planning and hypothesis generation; ② GeneDataAgent for tool-mediated H5 multi-omics network data retrieval (13,627 genes, 504,378 edges); ③ LLMFabAgent for direct gene-disease probability prediction from LLM training knowledge; ④ GeneAnalysisAgent for biological interpretation against the COSMIC cancer gene census; and ⑤ LiteratureAgent for real-time validation via PubMed, MyGene, and Open Targets public APIs. To suppress hallucination, we designed a three-layer defense: mandatory tool invocation (tool_call_required), grounded output validation against tool-returned data, and cross-validation between LLM knowledge and real API evidence.

**Results**: The framework was validated on the CPDB human gene interaction network. LLMFabAgent predicted disease association probabilities for 15 genes (TP53: 0.99, KRAS: 0.99, BRCA1: 0.99, EGFR: 0.98), with 10 high-confidence predictions (67%) overlapping the COSMIC Cancer Gene Census. Real-time PubMed API cross-validation confirmed all 10 high-confidence predictions (EGFR: 64,359 publications, MYC: 38,816, TP53: 33,950, KRAS: 30,259), while 5 low-confidence genes (TTN, AHNAK, etc.) were correctly flagged as lacking literature support. GeneDataAgent successfully retrieved real network statistics from three H5 datasets (CPDB, STRINGdb, PCNET), eliminating data hallucination.

**Conclusion**: GeneAgent demonstrates the feasibility of LLMs as biomedical knowledge bases—when combined with rigorous tool-use constraints and real-time API cross-validation, LLMs can produce gene-disease association predictions that are both highly concordant with established biological knowledge and empirically verifiable. This framework establishes a reproducible paradigm for deploying LLMs in high-stakes biomedical applications, and its dual-engine architecture of "LLM knowledge prediction + real-time data validation" can be extended to drug repurposing, variant pathogenicity interpretation, and related tasks.

**Keywords**: Large language model; multi-agent system; gene-disease association prediction; hallucination mitigation; cross-validation; PubMed API; biomedical knowledge base
