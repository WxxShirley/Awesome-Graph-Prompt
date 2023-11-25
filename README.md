<h1 align="center"> Awesome-Graph-Prompt</h2>
<h5 align="center">A collection of AWESOME things about performing prompting on Graphs.</h5>

<h5 align="center">
  
![Awesome](https://awesome.re/badge.svg)   ![GitHub stars](https://img.shields.io/github/stars/WxxShirley/Awesome-Graph-Prompt.svg)

</h5>


Recently, the workflow of **"pre-train, fine-tune"** has been proved less effective and efficient when dealing with diverse downstream tasks on graph domain.
Inspired by the prompt learning in natural language processing (NLP) domain, the **"pre-train, prompt"** workflow has emerged as a promising solution. 

This repo aims to provide a curated list of research papers that explore the prompting on graphs. We will try to make this list updated frequently. If you found any error or any missed paper, please don't hesitate to open issues or pull requests.




## Table of Contents

- [Awesome-Graph-Prompt](#awesome-graph-prompt)
  - [Table of Contents](#table-of-contents)
  - [GNN Prompting Papers](#gnn-prompting-papers) 
  - [Application Papers](#application-papers)
     - [Social Networks](#social-networks)
     - [Recommender Systems](#recommender-systems)
     - [Knowledge Graph](#knowledge-graph)
     - [Biology](#biology)
     - [Others](#others)
  - [Ohter Resources](#other-resources)
     - [Open Source](#open-source)
     - [Datasets](#datasets)
     - [Blogs](#blogs)
  - [Contributing](#contributing)
  


## GNN Prompting Papers


1. **GPPT: Graph Pre-training and Prompt Tuning to Generalize Graph Neural Networks**.
   In **KDD'2022**, [[Paper](https://dl.acm.org/doi/10.1145/3534678.3539249 )]  [[Code](https://github.com/MingChen-Sun/GPPT)].

    ![](https://img.shields.io/badge/Encoder%3AGNN-green)  ![](https://img.shields.io/badge/Prompt%20as%20Tokens-red)  ![](https://img.shields.io/badge/Downstream%3A%20Node-yellow)

2. **SGL-PT: A Strong Graph Learner with Graph Prompt Tuning**.
   In **arXiv**, [[Paper](https://arxiv.org/abs/2302.12449)].

3. **GraphPrompt: Unifying Pre-Training and Downstream Tasks for Graph Neural Networks**.
   In **WWW'2023**, [[Paper](https://dl.acm.org/doi/10.1145/3543507.3583386 )] [[Code](https://github.com/Starlien95/GraphPrompt )].

   ![](https://img.shields.io/badge/Encoder%3AGNN-green)  ![](https://img.shields.io/badge/Prompt%20as%20Tokens-red) ![](https://img.shields.io/badge/Downstream%3A%20Node%2FGraph-yellow)

4. **All in One: Multi-Task Prompting for Graph Neural Networks**.
   In **KDD'2023** Best Paper Award 🌟, [[Paper](https://arxiv.org/abs/2307.01504 )]  [[Code](https://github.com/sheldonresearch/ProG)].

   ![](https://img.shields.io/badge/Encoder%3AGNN-green) ![](https://img.shields.io/badge/Prompt%20as%20Graphs-red) ![](https://img.shields.io/badge/Downstream%3A%20Node%2FEdge%2FGraph-yellow)

5. **Virtual Node Tuning for Few-shot Node Classification**.
   In **KDD'2023**, [[Paper](https://arxiv.org/abs/2306.06063)].

   ![](https://img.shields.io/badge/Encoder%3AGraph%20Transformer-green) ![](https://img.shields.io/badge/Prompt%20as%20Tokens-red) ![](https://img.shields.io/badge/Downstream%3A%20Node-yellow)

6. **PRODIGY: Enabling In-context Learning Over Graphs**.
   In **NeurIPS'2023** Spotlight 🌟, [[Paper](https://arxiv.org/abs/2305.12600)] [[Code](https://github.com/snap-stanford/prodigy )].

   ![](https://img.shields.io/badge/Encoder%3AGNN-green) ![](https://img.shields.io/badge/Prompt%20as%20Graphs-red) ![](https://img.shields.io/badge/Downstream%3A%20Node%2FEdge%2FGraph-yellow)

7. **Universal Prompt Tuning for Graph Neural Networks**.
   In **NeurIPS'2023**, [[Paper](https://arxiv.org/abs/2209.15240)].

   ![](https://img.shields.io/badge/Encoder%3AGNN-green)  ![](https://img.shields.io/badge/Prompt%20as%20Tokens-red) ![](https://img.shields.io/badge/Downstream%3A%20Graph-yellow)

8. **Deep Prompt Tuning for Graph Transformers**.
   In **arXiv**, [[Paper](https://arxiv.org/abs/2309.10131)].

   ![](https://img.shields.io/badge/Encoder%3AGraph%20Transformer-green) ![](https://img.shields.io/badge/Prompt%20as%20Tokens-red) ![](https://img.shields.io/badge/Downstream%3A%20Graph-yellow)

9. **Prompt Tuning for Multi-View Graph Contrastive Learning**.
   In **arXiv**, [[Paper](https://arxiv.org/abs/2310.10362)].

10. **ULTRA-DP:Unifying Graph Pre-training with Multi-task Graph Dual Prompt**.
    In **arXiv**, [[Paper](https://arxiv.org/abs/2310.14845)].

    ![](https://img.shields.io/badge/Encoder%3AGNN-green)  ![](https://img.shields.io/badge/Prompt%20as%20Tokens-red)  ![](https://img.shields.io/badge/Downstream%3A%20Node-yellow).

11. **HetGPT: Harnessing the Power of Prompt Tuning in Pre-Trained Heterogeneous Graph Neural Networks**.
    In **arXiv**, [[Paper](https://arxiv.org/abs/2310.15318)].

    ![](https://img.shields.io/badge/Encoder%3AGNN-green)  ![](https://img.shields.io/badge/Prompt%20as%20Tokens-red)  ![](https://img.shields.io/badge/Downstream%3A%20Node-yellow).

12. **Enhancing Graph Neural Networks with Structure-Based Prompt**.
    In **arXiv**, [[Paper](https://arxiv.org/abs/2310.17394)].

    ![](https://img.shields.io/badge/Encoder%3AGNN-green)  ![](https://img.shields.io/badge/Prompt%20as%20Tokens-red) ![](https://img.shields.io/badge/Downstream%3A%20Node%2FGraph-yellow)




## Application Papers


### Social Networks
1. **Prompt-and-Align: Prompt-Based Social Alignment for Few-Shot Fake News Detection**.
   In **CIKM'2023**, [[Paper](https://arxiv.org/pdf/2309.16424.pdf )] [[Code](https://github.com/jiayingwu19/Prompt-and-Align)]. `Fake News Detection`
2. **Voucher Abuse Detection with Prompt-based Fine-tuning on Graph Neural Networks**.
   In **CIKM'2023**, [[Paper](https://arxiv.org/abs/2308.10028 )]. `Fraud Detection`


### Recommender Systems
1. **Contrastive Graph Prompt-tuning for Cross-domain Recommendation**.
   In **TOIS'2023**, [[Paper](https://arxiv.org/pdf/2308.10685.pdf )]. `Cross-domain Recommendation`
2. **An Empirical Study Towards Prompt-Tuning for Graph Contrastive Pre-Training in Recommendations**.
   In **NeurIPS'2023**, [[Paper](https://openreview.net/pdf?id=XyAP8ScqLV)] [[Code](https://github.com/Haoran-Young/CPTPP )]. `General Recommendation`
3. **Motif-Based Prompt Learning for Universal Cross-Domain Recommendation**.
   In **WSDM'2024**, [[Paper](https://arxiv.org/abs/2310.13303)]. `Cross-domain Recommendation`

### Knowledge Graph
1. **Structure Pretraining and Prompt Tuning for Knowledge Graph Transfer**.
   In **WWW'2023**, [[Paper](https://arxiv.org/pdf/2303.03922.pdf )] [[Code](https://github.com/zjukg/KGTransformer )]. 
2. **Graph Neural Prompting with Large Language Models**.
   In **arXiv**, [[Paper](https://arxiv.org/pdf/2309.15427.pdf)].
3. **Knowledge Graph Prompting for Multi-Document Question Answering**.
   In **arXiv**, [[Paper](https://arxiv.org/abs/2308.11730 )] [[Code](https://github.com/YuWVandy/KG-LLM-MDQA )].

### Biology
1. **GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning**.
   In **NeurIPS'2023**, [[Paper](https://arxiv.org/pdf/2306.13089.pdf)] [[Code](https://github.com/zhao-ht/GIMLET )].
2. **MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter**.
   In **EMNLP'2023**, [[Paper](http://arxiv.org/abs/2310.12798)] [[Code](https://github.com/acharkq/MolCA)].


### Others
1. **A Data-centric Framework to Endow Graph Neural Networks with Out-Of-Distribution Detection Ability**.
   In **KDD'2023**, [[Paper](https://dl.acm.org/doi/abs/10.1145/3580305.3599244)] [[Code](https://github.com/BUPT-GAMMA/AAGOD )]. `OOD Detection`




## Other Resources

### Open Source
* **ProG: A Unified Library for Graph Prompting** [[Website](https://graphprompt.github.io/)] [[Code](https://github.com/sheldonresearch/ProG)]
  
  ProG (Prompt Graph) is a library built upon PyTorch to easily conduct single or multiple task prompting for a pre-trained Graph Neural Networks (GNNs).

### Datasets

Datasets that are commonly used in GNN prompting papers.

|     Dataset      |     Category      | \#Graph | \#Node (Avg.) | \#Edge (Avg.) | \#Feature | \#Class |
| :--------------: | :---------------: | :-----: | :-----------: | :-----------: | :-------: | :-----: |
|       Cora       | Citation Network  |    1    |     2708      |     5429      |   1433    |    7    |
|     CoraFull     | Citation Network  |    1    |     19793     |     63421     |   8710    |   70    |
|     Citeseer     | Citation Network  |    1    |     3327      |     4732      |   3703    |    6    |
|      DBLP        | Citation Network  |    1    |     17716     |     105734    |   1639    |    4    |
|      Pubmed      | Citation Network  |    1    |     19717     |     44338     |    500    |    3    |
|   Coauthor-CS    | Citation Network  |    1    |     18333     |     81894     |   6805    |   15    |
| Coauthor-Physics | Citation Network  |    1    |     34493     |    247962     |   8415    |    5    |
|    ogbn-arxiv    | Citation Network  |    1    |    169343     |    1166243    |    128    |   40    |
| Amazon-Computers |  Purchase Network |    1    |     13752     |    245861     |    767    |   10    |
|   Amazon-Photo   |  Purchase Network |    1    |     7650      |    119081     |    745    |    8    |
|  ogbn-products   |  Purchase Network |    1    |    2449029    |   61859140    |    100    |   47    |
|     Wiki-CS      |      Web Link     |    1    |     11701     |    216123     |    300    |   10    |
|    FB15K237      | Knowledge Graph   |    1    |    14541      |    310116     |     -     |   237   |
|    WN18RR        | Knowledge Graph   |    1    |    40943      |     93003     |     -     |    11   |
|      Reddit      |  Social Network   |    1    |    232965     |   11606919    |    602    |   41    |
|      Flickr      |  Social Network   |    1    |     89250     |    899756     |     500   |    7    |
|     PROTEINS     | Protein Networks  |  1113   |     39.06     |     72.82     |     1     |    2    |
|       COX2       |  Molecule Graphs  |   467   |     41.22     |     43.45     |     3     |    2    |
|     ENZYMES      |  Molecule Graphs  |   600   |     32.63     |     62.14     |     18    |    6    |
|      MUTAG       |  Molecule Graphs  |   188   |     17.93     |     19.79     |     7     |    2    |
|       MUV        |  Molecule Graphs  |  93087  |     24.23     |     26.28     |     -     |   17    |
|       HIV        |  Molecule Graphs  |  41127  |     25.53     |     27.48     |     -     |    2    |
|      SIDER       |  Molecule Graphs  |  1427   |     33.64     |     35.36     |     -     |   27    |


### Blogs
* A Chinese Blog on Graph Prompting (including GPPT, GraphPrompt, All in One, etc) [[Link](https://mp.weixin.qq.com/s/_Khx87cdN6RGiOGkiUjV8Q)]


## Contributing
👍 Contributions to this repository are welcome! 

If you have come across relevant resources, feel free to open an issue or submit a pull request.
```
- paper_name (***journal***) [[Paper](link)] [[Code](link)]
```
