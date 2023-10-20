# Awesome-Graph-Prompt [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)


A collection of AWESOME things about performing **Pre-training and Prompt on Graphs**.

Recently, the workflow of **"pre-training and fine-tuning"** has been proved less effective and efficient when applied to diverse graph downstream tasks.
Inspired by the prompt learning in natural language processing (NLP), the **"pre-training and prompting"** workflow has emerged as a promising solution. 

This repo aims to provide a curated list of research papers that explore the prompting on graphs.


## Table of Contents

- [Awesome-Graph-Prompt](#awesome-graph-prompt-awesomehttpsawesomerebadgesvghttpsawesomere)
  - [Table of Contents](#table-of-contents)
  - [Survey](#survey)
  - [GNN Prompting Papers](#gnn-prompting-papers) 
  - [Application Papers](#application-papers)
     - [Large Language Models(LLMs)](#large-language-models)
     - [Recommender Systems](#recommender-systems)
     - [Text Attributed Graphs](#text-attributed-graphs )
     - [Question Answering](#question-answering)
     - [Fake News Detection](#fake-news-detection)
     - [Fraud Detection](#fraud-detection)
     - [OOD Detection](#ood-detection)
  - [Ohter Resources](#other-resources)
     - [Open Source](#open-source)
     - [Datasets](#datasets)
     - [Blogs](#blogs)
  - [Contributing](#contributing)
  



## Survey

* A Survey of Graph Prompting Methods: Techniques, Applications, and Challenges (*May 2023, arXiv*) [[Paper](https://arxiv.org/abs/2303.07275)]



## GNN Prompting Papers

* Prompt Tuning for Multi-View Graph Contrastive Learning (***October 2023, arXiv***) [[Paper](https://arxiv.org/abs/2310.10362)]
* GraphControl: Adding Conditional Control to Universal Graph Pre-trained Models for Graph Domain Transfer Learning (***October 2023, arXiv***) [[Paper](https://arxiv.org/abs/2310.07365)]
* Deep Prompt Tuning for Graph Transformers (***September 2023, arXiv***) [[Paper](https://arxiv.org/abs/2309.10131))
* Universal Prompt Tuning for Graph Neural Networks (***NeurIPS'2023***) [[Paper](https://arxiv.org/abs/2209.15240)]
* Virtual Node Tuning for Few-shot Node Classification (***KDD'2023***) [[Paper](https://arxiv.org/abs/2306.06063)]
* All in One: Multi-Task Prompting for Graph Neural Networks (***KDD'2023 Best Paper Award 🌟***) [[Paper](https://arxiv.org/abs/2307.01504 )]  [[Code](https://github.com/sheldonresearch/ProG)]
* PRODIGY: Enabling In-context Learning Over Graphs (***May 2023, arXiv***) [[Paper](https://arxiv.org/abs/2305.12600)]
* GraphPrompt: Unifying Pre-Training and Downstream Tasks for Graph Neural Networks (***WWW'2023***) [[Paper](https://dl.acm.org/doi/10.1145/3543507.3583386 )] [[Code](https://github.com/Starlien95/GraphPrompt )]
* SGL-PT: A Strong Graph Learner with Graph Prompt Tuning (***Feb 2023, arXiv***) [[Paper](https://arxiv.org/abs/2302.12449)]
* GPPT: Graph Pre-training and Prompt Tuning to Generalize Graph Neural Networks (***KDD'2022***) [[Paper](https://dl.acm.org/doi/10.1145/3534678.3539249 )]  [[Code](https://github.com/MingChen-Sun/GPPT)]

 
A Summary of Methodology Details 

|       Model     |     Pre-training Method          | Downstream Tasks | Prompt Design                             | Pub   |
| :-------------: | :------------------------------: | :--------------: | :---------------------------------------: | :---: |
|    All in One   |For All (work as a plug-and-play) |  Node/Edge/Graph |   Prompt tokens with learnable structures | KDD23 |
|      PGCL       |            Customozied           |  Node/Edge/Graph |    Prompt tokens                          | arXiv23/10|
|     GraphPrompt |         Link Prediction          |    Node/Graph    |    Prompt token                           | WWW23 |
|     GPF-plus    |For All (work as a plug-and-play) |         Graph    |    Prompt tokens                          | NIPS23|
|     GPF         |For All (work as a plug-and-play) |         Graph    |    Prompt token                           | NIPS23|
|      GPPT       |          Link Prediction         |    Node          |    Prompt tokens                          | KDD22 |




## Application Papers

### Large Language Models
> A relatively rough classification. These papers contruct *text* prompt and exploit LLM to solove graph-domain downstream tasks.

* Talk like a graph: Encoding graphs for large language models (***October 2023, arXiv***) [[Paper](https://arxiv.org/abs/2310.04560)]
* Graph Neural Prompting with Large Language Models (***September 2023, arXiv***)  [[Paper](https://arxiv.org/pdf/2309.15427.pdf)]
* One for All: Towards Training One Graph Model for All Classification Tasks (***September 2023, arXiv***) [[Paper](https://arxiv.org/abs/2310.00149 )] [[Code](https://github.com/LechengKong/OneForAll)]
* Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs (***Augest 2023, arXiv***) [[Paper](https://arxiv.org/abs/2307.03393)]

### Recommender Systems 

* An Empirical Study Towards Prompt-Tuning for Graph Contrastive Pre-Training in Recommendations (***NeurIPS'2023***)
* Contrastive Graph Prompt-tuning for Cross-domain Recommendation (***August 2023, arXiv***) [[Paper](https://arxiv.org/pdf/2308.10685.pdf )]

### Text Attributed Graphs 

* Prompt-based Node Feature Extractor for Few-shot Learning on Text-Attributed Graphs (***September 2023, arXiv***) [[Paper](https://arxiv.org/abs/2309.02848 )]
* Prompt-Based Zero- and Few-Shot Node Classification: A Multimodal Approach (***July 2023, arXiv***) [[Paper](https://arxiv.org/abs/2307.11572 )]
* Prompt Tuning on Graph-augmented Low-resource Text Classification (***July 2023, arXiv***) [[Paper](https://arxiv.org/abs/2307.10230 )] [[Code](https://github.com/WenZhihao666/G2P2-conditional )]
* Augmenting Low-Resource Text Classification with Graph-Grounded Pre-training and Prompting (***SIGIR'2023***) [[Paper](https://arxiv.org/abs/2305.03324 )] [[Code](https://github.com/WenZhihao666/G2P2 )]

### Question Answering

* Knowledge Graph Prompting for Multi-Document Question Answering (***August 2023, arXiv***) [[Paper](https://arxiv.org/abs/2308.11730 )] [[Code](https://github.com/YuWVandy/KG-LLM-MDQA )]

### Fake News Detection 

* Prompt-and-Align: Prompt-Based Social Alignment for Few-Shot Fake News Detection (***CIKM'2023***) [[Paper](https://arxiv.org/pdf/2309.16424.pdf )] [[Code](https://github.com/jiayingwu19/Prompt-and-Align)]

### Fraud Detection

* Voucher Abuse Detection with Prompt-based Fine-tuning on Graph Neural Networks (***CIKM'2023***) [[Paper](https://arxiv.org/abs/2308.10028 )]
  

### OOD Detection 

* A Data-centric Framework to Endow Graph Neural Networks with Out-Of-Distribution Detection Ability (***KDD'2023***) [[Paper](http://shichuan.org/doc/150.pdf)] [[Code](https://github.com/BUPT-GAMMA/AAGOD )]




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
