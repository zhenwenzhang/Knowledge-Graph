# Knowledge-Graph Papers

### Can We Predict New Facts with Open Knowledge Graph Embeddings? A Benchmark for Open Link Prediction[[ACL 2020](https://www.aclweb.org/anthology/P19-1139.pdf)]

*Houyu Zhang, Zhenghao Liu, Chenyan Xiong, Zhiyuan Liu*

Brown University,Tsinghua University,Microsoft Research AI

**Abstract** Human conversations naturally evolve around related concepts and hop to distant concepts. This paper presents a new conversation generation model, ConceptFlow, which leverages commonsense knowledge graphs to explicitly model conversation flows. By grounding conversations to the concept space, ConceptFlow represents the potential conversation flow as traverses in the concept space along commonsense relations. The traverse is guided by graph attentions in the concept graph, moving towards more meaningful directions in the concept space, in order to generate more semantic and informative responses. Experiments on Reddit conversations demonstrate ConceptFlow’s effectiveness over previous knowledge-aware conversation models and GPT-2 based models while using 70% fewer parameters, confirming the advantage of explicit modeling conversation structures. All source codes of this work are available at https://github.com/thunlp/ConceptFlow.


### Grounded Conversation Generation as Guided Traverses in Commonsense Knowledge Graphs[[ACL 2020](https://www.aclweb.org/anthology/P19-1139.pdf)]

*Houyu Zhang, Zhenghao Liu, Chenyan Xiong, Zhiyuan Liu*

Brown University,Tsinghua University,Microsoft Research AI

**Abstract** Human conversations naturally evolve around related concepts and hop to distant concepts. This paper presents a new conversation generation model, ConceptFlow, which leverages commonsense knowledge graphs to explicitly model conversation flows. By grounding conversations to the concept space, ConceptFlow represents the potential conversation flow as traverses in the concept space along commonsense relations. The traverse is guided by graph attentions in the concept graph, moving towards more meaningful directions in the concept space, in order to generate more semantic and informative responses. Experiments on Reddit conversations demonstrate ConceptFlow’s effectiveness over previous knowledge-aware conversation models and GPT-2 based models while using 70% fewer parameters, confirming the advantage of explicit modeling conversation structures. All source codes of this work are available at https://github.com/thunlp/ConceptFlow.

### ERNIE: Enhanced Language Representation with Informative Entities[[ACL 2019](https://www.aclweb.org/anthology/P19-1139.pdf)]

*Zhengyan Zhang, Xu Han, Zhiyuan Liu, Xin Jiang, Maosong Sun, Qun Liu*

Tsinghua University,Huawei Noah’s Ark Lab

**Abstract** Neural language representation models such as BERT pre-trained on large-scale corpora can well capture rich semantic patterns from plain text, and be fine-tuned to consistently improve the performance of various NLP tasks. However, the existing pre-trained language models rarely consider incorporating knowledge graphs (KGs), which can provide rich structured knowledge facts for better language understanding. We argue that informative entities in KGs can enhance language representation with external knowledge. In this paper, we utilize both large-scale textual corpora and KGs to train an enhanced language representation model (ERNIE), which can take full advantage of lexical, syntactic, and knowledge information simultaneously. The experimental results have demonstrated that ERNIE achieves significant improvements on various knowledge-driven tasks, and meanwhile is comparable with the state-of-the-art model BERT on other common NLP tasks. The source code of this paper can be obtained from https://github.com/thunlp/ERNIE.


### K-BERT: Enabling Language Representation with Knowledge Graph[[AAAI 2020](https://arxiv.org/pdf/1909.07606v1.pdf)]

*Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Qi Ju, Haotang Deng and Ping Wang*

Peking University,Tencent Research,Beijing Normal University

**Abstract** Pre-trained language representation models, such as BERT,capture a general language representation from large-scalecorpora, but lack domain-specific knowledge. When readinga domain text, experts make inferences with relevantknowledge. For machines to achieve this capability, we proposea knowledge-enabled language representation model(K-BERT) with knowledge graphs (KGs), in which triplesare injected into the sentences as domain knowledge. However,too much knowledge incorporation may divert the sentencefrom its correct meaning, which is called knowledgenoise (KN) issue. To overcome KN, K-BERT introduces softpositionand visible matrix to limit the impact of knowledge.K-BERT can easily inject domain knowledge into the modelsby equipped with a KG without pre-training by-self becauseit is capable of loading model parameters from the pretrainedBERT. Our investigation reveals promising results intwelve NLP tasks. Especially in domain-specific tasks (includingfinance, law, and medicine), K-BERT significantlyoutperforms BERT, which demonstrates that K-BERT is anexcellent choice for solving the knowledge-driven problemsthat require experts.


### KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation[[PrePrint](https://arxiv.org/pdf/1911.06136.pdf)]

*Xiaozhi Wang, Tianyu Gao, Zhaocheng Zhu, Zhiyuan Liu, Juanzi Li, Jian Tang*

Tsinghua University,CIFAR AI Research Chair

**Abstract** Pre-trained language representation models(PLMs) cannot well capture factual knowledgefrom text. In contrast, knowledge embedding(KE) methods can effectively representthe relational facts in knowledge graphs(KGs) with informative entity embeddings,but conventional KE models do not utilizethe rich text data. In this paper, we proposea unified model for Knowledge Embeddingand Pre-trained LanguagE Representation(KEPLER), which can not only better integratefactual knowledge into PLMs but alsoeffectively learn KE through the abundantinformation in text. In KEPLER, we encodetextual descriptions of entities with a PLMas their embeddings, and then jointly optimizethe KE and language modeling objectives.Experimental results show that KEPLERachieves state-of-the-art performanceon various NLP tasks, and also works remarkablywell as an inductive KE modelon the link prediction task. Furthermore,for pre-training KEPLER and evaluating theKE performance, we construct Wikidata5M,a large-scale KG dataset with aligned entitydescriptions, and benchmark state-ofthe-art KE methods on it. It shall serveas a new KE benchmark and facilitate theresearch on large KG, inductive KE, andKG with text. The dataset can be obtainedfrom https://deepgraphlearning.github.io/project/wikidata5m.


### CoLAKE: Contextualized Language and Knowledge Embedding[[COLING 2020](https://arxiv.org/pdf/2010.00309.pdf)]

*Tianxiang Sun, Yunfan Shao, Xipeng Qiu, Qipeng Guo, Yaru Hu, Xuanjing Huang, Zheng Zhang*

Fudan University,Amazon Shanghai AI Lab

**Abstract** With the emerging branch of incorporating factual knowledge into pre-trained language models such as BERT, most existing models consider shallow, static, and separately pre-trained entity embeddings, which limits the performance gains of these models. Few works explore the potential of deep contextualized knowledge representation when injecting knowledge. In this paper, we propose the Contextualized Language and Knowledge Embedding (CoLAKE), which jointly learns contextualized representation for both language and knowledge with the extended MLM objective. Instead of injecting only entity embeddings, CoLAKE extracts the knowledge context of an entity from large-scale knowledge bases. To handle the heterogeneity of knowledge context and language context, we integrate them in a unified data structure, word-knowledge graph (WK graph). CoLAKE is pre-trained on large-scale WK graphs with the modified Transformer encoder. We conduct experiments on knowledge-driven tasks, knowledge probing tasks, and language understanding tasks. Experimental results show that CoLAKE outperforms previous counterparts on most of the tasks. Besides, CoLAKE achieves surprisingly high performance on our synthetic task called word-knowledge graph completion, which shows the superiority of simultaneously contextualizing language and knowledge representation.


### Exploiting Structured Knowledge in Text via Graph-Guided Representation Learning[[ICLR 2020](https://arxiv.org/pdf/2010.00309.pdf)]

*Tao Shen, Yi Mao, Pengcheng He, Guodong Long, Adam Trischler, Weizhu Chen*

University of Technology Sydney,Microsoft Dynamics 365 AI

**Abstract** In this work, we aim at equipping pre-trained language models with structured knowledge. We present two self-supervised tasks learning over raw text with the guidance from knowledge graphs. Building upon entity-level masked language models, our first contribution is an entity masking scheme that exploits relational knowledge underlying the text. This is fulfilled by using a linked knowledge graph to select informative entities and then masking their mentions. In addition we use knowledge graphs to obtain distractors for the masked entities, and propose a novel distractor-suppressed ranking objective which is optimized jointly with masked language model. In contrast to existing paradigms, our approach uses knowledge graphs implicitly, only during pre-training, to inject language models with structured knowledge via learning from raw text. It is more efficient than retrieval-based methods that perform entity linking and integration during finetuning and inference, and generalizes more effectively than the methods that directly learn from concatenated graph triples. Experiments show that our proposed model achieves improved performance on five benchmark datasets, including question answering and knowledge base completion tasks.


### K-ADAPTER: Infusing Knowledge into Pre-Trained Models with Adapters[[ICLR 2021](https://arxiv.org/pdf/2010.00309.pdf)]

*Ruize Wang, Duyu Tang, Nan Duan, Zhongyu Wei, Xuanjing Huang, Jianshu ji, Guihong Cao, Daxin Jiang, Ming Zhou*

Fudan University,Microsoft Research Asia

**Abstract** We study the problem of injecting knowledge into large pre-trained models like BERT and RoBERTa. Existing methods typically update the original parameters of pre-trained models when injecting knowledge. However, when multiple kinds of knowledge are injected, they may suffer from catastrophic forgetting. To address this, we propose K-Adapter, which remains the original parameters of the pre-trained model fixed and supports continual knowledge infusion. Taking RoBERTa as the pre-trained model, K-Adapter has a neural adapter for each kind of infused knowledge, like a plug-in connected to RoBERTa. There is no information flow between different adapters, thus different adapters are efficiently trained in a distributed way. We inject two kinds of knowledge, including factual knowledge obtained from automatically aligned text-triplets on Wikipedia and Wikidata, and linguistic knowledge obtained from dependency parsing. Results on three knowledge-driven tasks (total six datasets) including relation classification, entity typing and question answering demonstrate that each adapter improves the performance, and the combination of both adapters brings further improvements. Probing experiments further indicate that K-Adapter captures richer factual and commonsense knowledge than RoBERTa.


### Integrating Graph Contextualized Knowledge into Pre-trained Language Models[[AAAI 2020](https://arxiv.org/pdf/1912.00147.pdf)]

*Bin He, Di Zhou, Jinghui Xiao, Xin Jiang, Qun Liu, Nicholas Jing Yuan, Tong Xu*

Huawei Noahs Ark Lab, University of Science and Technology of China

**Abstract** Complex node interactions are common in knowledge graphs,and these interactions also contain rich knowledge information.However, traditional methods usually treat a triple asa training unit during the knowledge representation learning(KRL) procedure, neglecting contextualized informationof the nodes in knowledge graphs (KGs). We generalize themodeling object to a very general form, which theoreticallysupports any subgraph extracted from the knowledge graph,and these subgraphs are fed into a novel transformer-basedmodel to learn the knowledge embeddings. To broaden usagescenarios of knowledge, pre-trained language models areutilized to build a model that incorporates the learned knowledgerepresentations. Experimental results demonstrate thatour model achieves the state-of-the-art performance on severalmedical NLP tasks, and improvement above TransE indicatesthat our KRL method captures the graph contextualizedinformation effectively.


### JAKET: Joint Pre-training of Knowledge Graph and Language Understanding[[PrePrint](https://arxiv.org/pdf/2010.00796.pdf)s]

*Donghan Yu, Chenguang Zhu, Yiming Yang, Michael Zeng*

Carnegie Mellon University, Microsoft Cognitive Services Research Group

**Abstract** Knowledge graphs (KGs) contain rich information about world knowledge, entitiesand relations. Thus, they can be great supplements to existing pre-trainedlanguage models. However, it remains a challenge to efficiently integrate informationfrom KG into language modeling. And the understanding of a knowledgegraph requires related context. We propose a novel joint pre-training framework,JAKET, to model both the knowledge graph and language. The knowledge moduleand language module provide essential information to mutually assist eachother: the knowledge module produces embeddings for entities in text while thelanguage module generates context-aware initial embeddings for entities and relationsin the graph. Our design enables the pre-trained model to easily adaptto unseen knowledge graphs in new domains. Experimental results on severalknowledge-aware NLP tasks show that our proposed framework achieves superiorperformance by effectively leveraging knowledge in language understanding.
