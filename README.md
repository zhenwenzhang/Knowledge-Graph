# Knowledge-Graph Papers

### Benchmark and Best Practices for Biomedical Knowledge Graph Embeddings[[ACL 2020](https://www.aclweb.org/anthology/2020.bionlp-1.18.pdf)]

*David Chang, Ivana Balažević, Carl Allen, Daniel Chawla, Cynthia Brandt, Andrew Taylor*

Yale University,University of Edinburgh

**Abstract** Much of biomedical and healthcare data is encoded in discrete, symbolic form such as text and medical codes. There is a wealth of expert-curated biomedical domain knowledge stored in knowledge bases and ontologies, but the lack of reliable methods for learning knowledge representation has limited their usefulness in machine learning applications. While text-based representation learning has significantly improved in recent years through advances in natural language processing, attempts to learn biomedical concept embeddings so far have been lacking. A recent family of models called knowledge graph embeddings have shown promising results on general domain knowledge graphs, and we explore their capabilities in the biomedical domain. We train several state-of-the-art knowledge graph embedding models on the SNOMED-CT knowledge graph, provide a benchmark with comparison to existing methods and in-depth discussion on best practices, and make a case for the importance of leveraging the multi-relational nature of knowledge graphs for learning biomedical knowledge representation. The embeddings, code, and materials will be made available to the community.


### Low-Dimensional Hyperbolic Knowledge Graph Embeddings[[ACL 2020](https://www.aclweb.org/anthology/2020.acl-main.617.pdf)]

*Ines Chami, Adva Wolf, Da-Cheng Juan, Frederic Sala, Sujith Ravi, Christopher Ré*

Stanford University,Google Research,Amazon Alexa

**Abstract** Knowledge graph (KG) embeddings learn low- dimensional representations of entities and relations to predict missing facts. KGs often exhibit hierarchical and logical patterns which must be preserved in the embedding space. For hierarchical data, hyperbolic embedding methods have shown promise for high-fidelity and parsimonious representations. However, existing hyperbolic embedding methods do not account for the rich logical patterns in KGs. In this work, we introduce a class of hyperbolic KG embedding models that simultaneously capture hierarchical and logical patterns. Our approach combines hyperbolic reflections and rotations with attention to model complex relational patterns. Experimental results on standard KG benchmarks show that our method improves over previous Euclidean- and hyperbolic-based efforts by up to 6.1% in mean reciprocal rank (MRR) in low dimensions. Furthermore, we observe that different geometric transformations capture different types of relations while attention- based transformations generalize to multiple relations. In high dimensions, our approach yields new state-of-the-art MRRs of 49.6% on WN18RR and 57.7% on YAGO3-10.


### Connecting Embeddings for Knowledge Graph Entity Typing[[ACL 2020](https://www.aclweb.org/anthology/2020.acl-main.572.pdf)]

*Yu Zhao, Anxiang Zhang, Ruobing Xie, Kang Liu, Xiaojie Wang*

Southwestern University of Finance and Economics,Carnegie Mellon University,Tencent,Chinese Academy of Sciences,University of Chinese Academy of Sciences,Beijing University of Posts and Telecommunications

**Abstract** Knowledge graph (KG) entity typing aims at inferring possible missing entity type instances in KG, which is a very significant but still under-explored subtask of knowledge graph completion. In this paper, we propose a novel approach for KG entity typing which is trained by jointly utilizing local typing knowledge from existing entity type assertions and global triple knowledge in KGs. Specifically, we present two distinct knowledge-driven effective mechanisms of entity type inference. Accordingly, we build two novel embedding models to realize the mechanisms. Afterward, a joint model via connecting them is used to infer missing entity type instances, which favors inferences that agree with both entity type instances and triple knowledge in KGs. Experimental results on two real-world datasets (Freebase and YAGO) demonstrate the effectiveness of our proposed mechanisms and models for improving KG entity typing. The source code and data of this paper can be obtained from: https://github.com/Adam1679/ConnectE .


### ReInceptionE: Relation-Aware Inception Network with Joint Local-Global Structural Information for Knowledge Graph Embedding[[ACL 2020](https://www.aclweb.org/anthology/2020.acl-main.526.pdf)]

*Zhiwen Xie, Guangyou Zhou, Jin Liu, Jimmy Xiangji Huang*

Wuhan University,Central China Normal University,York University

**Abstract** The goal of Knowledge graph embedding (KGE) is to learn how to represent the low dimensional vectors for entities and relations based on the observed triples. The conventional shallow models are limited to their expressiveness. ConvE (Dettmers et al., 2018) takes advantage of CNN and improves the expressive power with parameter efficient operators by increasing the interactions between head and relation embeddings. However, there is no structural information in the embedding space of ConvE, and the performance is still limited by the number of interactions. The recent KBGAT (Nathani et al., 2019) provides another way to learn embeddings by adaptively utilizing structural information. In this paper, we take the benefits of ConvE and KBGAT together and propose a Relation-aware Inception network with joint local-global structural information for knowledge graph Embedding (ReInceptionE). Specifically, we first explore the Inception network to learn query embedding, which aims to further increase the interactions between head and relation embeddings. Then, we propose to use a relation-aware attention mechanism to enrich the query embedding with the local neighborhood and global entity information. Experimental results on both WN18RR and FB15k-237 datasets demonstrate that ReInceptionE achieves competitive performance compared with state-of-the-art methods.


### A Re-evaluation of Knowledge Graph Completion Methods[[ACL 2020](https://www.aclweb.org/anthology/2020.acl-main.489.pdf)]

*Zhiqing Sun, Shikhar Vashishth, Soumya Sanyal, Partha Talukdar, Yiming Yang*

Carnegie Mellon University,Indian Institute of Science

**Abstract** Knowledge Graph Completion (KGC) aims at automatically predicting missing links for large-scale knowledge graphs. A vast number of state-of-the-art KGC techniques have got published at top conferences in several research fields, including data mining, machine learning, and natural language processing. However, we notice that several recent papers report very high performance, which largely outperforms previous state-of-the-art methods. In this paper, we find that this can be attributed to the inappropriate evaluation protocol used by them and propose a simple evaluation protocol to address this problem. The proposed protocol is robust to handle bias in the model, which can substantially affect the final results. We conduct extensive experiments and report performance of several existing methods using our protocol. The reproducible code has been made publicly available.


### Knowledge Graph-Augmented Abstractive Summarization with Semantic-Driven Cloze Reward[[ACL 2020](https://www.aclweb.org/anthology/2020.acl-main.457.pdf)]

*Luyang Huang, Lingfei Wu, Lu Wang*

Northeastern University,IBM Research AI

**Abstract** Sequence-to-sequence models for abstractive summarization have been studied extensively, yet the generated summaries commonly suffer from fabricated content, and are often found to be near-extractive. We argue that, to address these issues, the summarizer should acquire semantic interpretation over input, e.g., via structured representation, to allow the generation of more informative summaries. In this paper, we present ASGARD, a novel framework for Abstractive Summarization with Graph-Augmentation and semantic-driven RewarD. We propose the use of dual encoders—a sequential document encoder and a graph-structured encoder—to maintain the global context and local characteristics of entities, complementing each other. We further design a reward based on a multiple choice cloze test to drive the model to better capture entity interactions. Results show that our models produce significantly higher ROUGE scores than a variant without knowledge graph as input on both New York Times and CNN/Daily Mail datasets. We also obtain better or comparable performance compared to systems that are fine-tuned from large pretrained language models. Human judges further rate our model outputs as more informative and containing fewer unfaithful errors.


### Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings[[ACL 2020](https://www.aclweb.org/anthology/2020.acl-main.412.pdf)]

*Apoorv Saxena, Aditay Tripathi, Partha Talukdar*

Indian Institute of Science, Bangalore

**Abstract** Knowledge Graphs (KG) are multi-relational graphs consisting of entities as nodes and relations among them as typed edges. Goal of the Question Answering over KG (KGQA) task is to answer natural language queries posed over the KG. Multi-hop KGQA requires reasoning over multiple edges of the KG to arrive at the right answer. KGs are often incomplete with many missing links, posing additional challenges for KGQA, especially for multi-hop KGQA. Recent research on multi-hop KGQA has attempted to handle KG sparsity using relevant external text, which isn’t always readily available. In a separate line of research, KG embedding methods have been proposed to reduce KG sparsity by performing missing link prediction. Such KG embedding methods, even though highly relevant, have not been explored for multi-hop KGQA so far. We fill this gap in this paper and propose EmbedKGQA. EmbedKGQA is particularly effective in performing multi-hop KGQA over sparse KGs. EmbedKGQA also relaxes the requirement of answer selection from a pre-specified neighborhood, a sub-optimal constraint enforced by previous multi-hop KGQA methods. Through extensive experiments on multiple benchmark datasets, we demonstrate EmbedKGQA’s effectiveness over other state-of-the-art baselines.

### SEEK: Segmented Embedding of Knowledge Graphs[[ACL 2020](https://www.aclweb.org/anthology/2020.acl-main.358.pdf)]

*Wentao Xu, Shun Zheng, Liang He, Bin Shao, Jian Yin, Tie-Yan Liu*

Sun Yat-sen University, Microsoft Research Asia

**Abstract** In recent years, knowledge graph embedding becomes a pretty hot research topic of artificial intelligence and plays increasingly vital roles in various downstream applications, such as recommendation and question answering. However, existing methods for knowledge graph embedding can not make a proper trade-off between the model complexity and the model expressiveness, which makes them still far from satisfactory. To mitigate this problem, we propose a lightweight modeling framework that can achieve highly competitive relational expressiveness without increasing the model complexity. Our framework focuses on the design of scoring functions and highlights two critical characteristics: 1) facilitating sufficient feature interactions; 2) preserving both symmetry and antisymmetry properties of relations. It is noteworthy that owing to the general and elegant design of scoring functions, our framework can incorporate many famous existing methods as special cases. Moreover, extensive experiments on public benchmarks demonstrate the efficiency and effectiveness of our framework. Source codes and data can be found at https://github.com/Wentao-Xu/SEEK.


### Breaking Through the 80% Glass Ceiling: Raising the State of the Art in Word Sense Disambiguation by Incorporating Knowledge Graph Information[[ACL 2020](https://www.aclweb.org/anthology/2020.acl-main.255.pdf)]

*Michele Bevilacqua, Roberto Navigli*

Sapienza University of Rome

**Abstract** Neural architectures are the current state of the art in Word Sense Disambiguation (WSD). However, they make limited use of the vast amount of relational information encoded in Lexical Knowledge Bases (LKB). We present Enhanced WSD Integrating Synset Embeddings and Relations (EWISER), a neural supervised architecture that is able to tap into this wealth of knowledge by embedding information from the LKB graph within the neural architecture, and to exploit pretrained synset embeddings, enabling the network to predict synsets that are not in the training set. As a result, we set a new state of the art on almost all the evaluation settings considered, also breaking through, for the first time, the 80% ceiling on the concatenation of all the standard all-words English WSD evaluation benchmarks. On multilingual all-words WSD, we report state-of-the-art results by training on nothing but English.


### Orthogonal Relation Transforms with Graph Context Modeling for Knowledge Graph Embedding[[ACL 2020](https://www.aclweb.org/anthology/2020.acl-main.241.pdf)]

*Yun Tang, Jing Huang, Guangtao Wang, Xiaodong He, Bowen Zhou*

JD AI Research

**Abstract** Distance-based knowledge graph embeddings have shown substantial improvement on the knowledge graph link prediction task, from TransE to the latest state-of-the-art RotatE. However, complex relations such as N-to-1, 1-to-N and N-to-N still remain challenging to predict. In this work, we propose a novel distance-based approach for knowledge graph link prediction. First, we extend the RotatE from 2D complex domain to high dimensional space with orthogonal transforms to model relations. The orthogonal transform embedding for relations keeps the capability for modeling symmetric/anti-symmetric, inverse and compositional relations while achieves better modeling capacity. Second, the graph context is integrated into distance scoring functions directly. Specifically, graph context is explicitly modeled via two directed context representations. Each node embedding in knowledge graph is augmented with two context representations, which are computed from the neighboring outgoing and incoming nodes/edges respectively. The proposed approach improves prediction accuracy on the difficult N-to-1, 1-to-N and N-to-N cases. Our experimental results show that it achieves state-of-the-art results on two common benchmarks FB15k-237 and WNRR-18, especially on FB15k-237 which has many high in-degree nodes.


### Knowledge Graph Embedding Compression[[ACL 2020](https://www.aclweb.org/anthology/2020.acl-main.238.pdf)]

*Mrinmaya Sachan*

Toyota Technological Institute at Chicago

**Abstract** Knowledge graph (KG) representation learning techniques that learn continuous embeddings of entities and relations in the KG have become popular in many AI applications. With a large KG, the embeddings consume a large amount of storage and memory. This is problematic and prohibits the deployment of these techniques in many real world settings. Thus, we propose an approach that compresses the KG embedding layer by representing each entity in the KG as a vector of discrete codes and then composes the embeddings from these codes. The approach can be trained end-to-end with simple modifications to any existing KG embedding technique. We evaluate the approach on various standard KG embedding evaluations and show that it achieves 50-1000x compression of embeddings with a minor loss in performance. The compressed embeddings also retain the ability to perform various reasoning tasks such as KG inference.


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
