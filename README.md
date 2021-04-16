# TWEAC: Transformer with Extendable QA Agent Classifiers

This repository contains the code for the paper [TWEAC: Transformer with Extendable QA Agent Classifiers](https://arxiv.org/abs/2104.07081).

Please use the following citation:

```
@article{geigle:2021:arxiv,
  author    = {Gregor Geigle and 
                Nils Reimers and 
                Andreas R{\"u}ckl{\'e} and
                Iryna Gurevych},
  title     = {TWEAC: Transformer with Extendable QA Agent Classifiers},
  journal   = {arXiv preprint},
  volume    = {abs/2104.07081},
  year      = {2021},
  url       = {http://arxiv.org/abs/2104.07081},
  archivePrefix = {arXiv},
  eprint    = {2104.07081}
}
```

> **Abstract:** 
> Question answering systems should help users to access knowledge on a broad range of topics and to answer a wide array of different questions. Most systems fall short of this expectation as they are only specialized in one particular setting, e.g., answering factual questions with Wikipedia data.
> To overcome this limitation, we propose composing multiple \emph{QA agents} within a meta-QA system. We argue that there exist a wide range of specialized QA agents in literature. Thus, we address the central research question of how to effectively and efficiently identify suitable QA agents for any given question.
> We study both supervised and unsupervised approaches to address this challenge, showing that \emph{TWEAC}---Transformer with Extendable Agent Classifiers---achieves the best performance overall with 94\% accuracy. We provide extensive insights on the scalability of TWEAC, demonstrating that it scales robustly to over 100 QA agents with each providing just 1000 examples of questions they can answer.

Contact person: Gregor Geigle, gregor.geigle@gmail.com

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.


## Structure
* `data`: our training and test data. See the data section below for more information about the format.
* `tweak`: our TWEAK model with training and evaluation code. 
* `elasticsearch`, `es_classifier`, `es_vec_classifier`, `faiss`, `faiss_classifier`:
code to setup and evaluate the BM25- and sentence-embedding-based models

Each directory contains additional READMEs and configs detailing the usage.

## Requirements & Installation
Python >=3.6 is recommended.   
TWEAK requires Pytorch (tested up to version 1.6.0) and Huggingface Transformers (tested up to version 4.1.1).
The other approaches which use Elasticsearch and Faiss require additional packages.
We include ``requirements.txt`` in each folder (which were generated with pipreqs) so you only have to install the necessary packages for each approach.

For Elasticsearch, we provide a Dockerfile for an Elasticsearch instance with the required plugin for embedding-based search (https://github.com/lior-k/fast-elasticsearch-vector-scoring).
We use Elasticsearch 7.5.0.

## Data
We include our processed data for our experiments in [data](data).
Each folder corresponds to one (or for Stackexchange and reddit multiple) agent(s).
The training, test and dev data for each agent is placed in their respective folder.
Each folder contains text files where each line represents one example question.

## Running the Experiments
We include READMEs in the folders [tweak](tweak/README.md), [es_classifier](es_classifier/README.md), 
[es_vec_classifier](es_vec_classifier/README.md) and [faiss_classifier](faiss_classifier/README.md) that detail how to 
run our experiments with the respective approach.

For TWEAK, we also detail how to train our proposed model and to use our proposed extension strategies.
