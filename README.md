# Knowledge-Graph Papers

**ERNIE: Enhanced Language Representation with Informative Entities**[ACL 2019](https://www.aclweb.org/anthology/P19-1139.pdf)

*Zhengyan Zhang, Xu Han, Zhiyuan Liu, Xin Jiang, Maosong Sun, Qun Liu*, Tsinghua University,Huawei Noah’s Ark Lab

**Abstract** Neural language representation models such as BERT pre-trained on large-scale corpora can well capture rich semantic patterns from plain text, and be fine-tuned to consistently improve the performance of various NLP tasks. However, the existing pre-trained language models rarely consider incorporating knowledge graphs (KGs), which can provide rich structured knowledge facts for better language understanding. We argue that informative entities in KGs can enhance language representation with external knowledge. In this paper, we utilize both large-scale textual corpora and KGs to train an enhanced language representation model (ERNIE), which can take full advantage of lexical, syntactic, and knowledge information simultaneously. The experimental results have demonstrated that ERNIE achieves significant improvements on various knowledge-driven tasks, and meanwhile is comparable with the state-of-the-art model BERT on other common NLP tasks. The source code of this paper can be obtained from https://github.com/thunlp/ERNIE.


- K-BERT: Enabling Language Representation with Knowledge Graph[AAAI 2020](https://arxiv.org/pdf/1909.07606v1.pdf)

*Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Qi Ju,*, Haotang Deng and Ping Wang*,Peking University,Tencent Research,Beijing Normal University

**Abstract** Pre-trained language representation models, such as BERT,capture a general language representation from large-scalecorpora, but lack domain-specific knowledge. When readinga domain text, experts make inferences with relevantknowledge. For machines to achieve this capability, we proposea knowledge-enabled language representation model(K-BERT) with knowledge graphs (KGs), in which triplesare injected into the sentences as domain knowledge. However,too much knowledge incorporation may divert the sentencefrom its correct meaning, which is called knowledgenoise (KN) issue. To overcome KN, K-BERT introduces softpositionand visible matrix to limit the impact of knowledge.K-BERT can easily inject domain knowledge into the modelsby equipped with a KG without pre-training by-self becauseit is capable of loading model parameters from the pretrainedBERT. Our investigation reveals promising results intwelve NLP tasks. Especially in domain-specific tasks (includingfinance, law, and medicine), K-BERT significantlyoutperforms BERT, which demonstrates that K-BERT is anexcellent choice for solving the knowledge-driven problemsthat require experts.


- KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation
