# MedRAG Toolkit

`MedRAG` a systematic toolkit for Retrieval-Augmented Generation (RAG) on medical question answering (QA). `MedRAG` is used to implement various RAG systems for the benchmark study on our [`MIRAGE`](https://github.com/Teddy-XiongGZ/MIRAGE) (Medical Information Retrieval-Augmented Generation Evaluation).

## Table of Contents

- [Introduction](#introduction)
- - [Corpus](#corpus)
- - [Retriever](#retriever)
- - [LLM](#llm)
- [Requirements](#requirements)
- [Usage](#usage)
- [Citation](#citation)

## Introduction

The following figure shows that MedRAG consists of three major components: Corpora, Retrievers, and LLMs.

<img src="figs/MedRAG.png" alt="Alt text" width="375"/>

### Corpus

For corpora used in MedRAG, we collect raw data from four different sources, including the commonly used [PubMed](https://pubmed.ncbi.nlm.nih.gov/) for all biomedical abstracts, [StatPearls](https://www.statpearls.com/) for clinical decision support, medical [Textbooks](https://github.com/jind11/MedQA) for domain-specific knowledge, and [Wikipedia](https://huggingface.co/datasets/wikipedia) for general knowledge. We also provide a MedCorp corpus by combining all four corpora, facilitating cross-source retrieval. Each corpus is chunked into short snippets.

| **Corpus**  | **#Doc.** | **#Snippets** | **Avg. L** | **Domain** |
|-------------|-----------|---------------|------------|------------|
| PubMed      | 23.9M     | 23.9M         | 296        | Biomed.    |
| StatPearls  | 9.3k      | 301.2k        | 119        | Clinics    |
| Textbooks   | 18        | 125.8k        | 182        | Medicine   |
| Wikipedia   | 6.5M      | 29.9M         | 162        | General    |
| MedCorp     | 30.4M     | 54.2M         | 221        | Mixed      |

(\#Doc.: numbers of raw documents; \#Snippets: numbers of snippets (chunks); Avg. L: average length of snippets.)

### Retriever

For the retrieval algorithms, we only select some representative ones in MedRAG, including a lexical retriever ([BM25](https://github.com/castorini/pyserini)), a general-domain semantic retriever ([Contriever](https://huggingface.co/facebook/contriever)), a scientific-domain retriever ([SPECTER](https://huggingface.co/allenai/specter)), and a biomedical-domain retriever ([MedCPT](https://huggingface.co/ncbi/MedCPT-Query-Encoder)).

| **Retriever** | **Type**   | **Size** | **Metric** | **Domain**   |
|---------------|------------|----------|------------|--------------|
| BM25          | Lexical    | --       | BM25       | General      |
| Contriever    | Semantic   | 110M     | IP         | General      |
| SPECTER       | Semantic   | 110M     | L2         | Scientific   |
| MedCPT        | Semantic   | 109M     | IP         | Biomed.      |

(IP: inner product; L2: L2 norm)

### LLM

We select several frequently used LLMs in MedRAG, including the commercial [GPT-3.5](https://platform.openai.com/) and [GPT-4](https://oai.azure.com/), the open-source [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) and [Llama2](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf), and the biomedical domain-specific [MEDITRON](https://huggingface.co/epfl-llm/meditron-70b) and [PMC-LLaMA](https://huggingface.co/axiong/PMC_LLaMA_13B).
Temperatures are set to 0 for deterministic outputs.

| **LLM**      | **Size** | **Context** | **Open** | **Domain** |
|--------------|----------|-------------|----------|------------|
| GPT-4        | N/A      | 32,768      | No       | General    |
| GPT-3.5      | N/A      | 16,384      | No       | General    |
| Mixtral      | 8×7B     | 32,768      | Yes      | General    |
| Llama2       | 70B      | 4,096       | Yes      | General    |
| MEDITRON     | 70B      | 4,096       | Yes      | Biomed.    |
| PMC-LLaMA    | 13B      | 2,048       | Yes      | Biomed.    |

(Context: context length of the LLM; Open: Open-source.)

## Requirements

- First, install PyTorch suitable for your system's CUDA version by following the [official instructions](https://pytorch.org/get-started/locally/) (2.1.1+cu121 in our case).

- Then, install the remaining requirements using: `pip install -r requirements.txt`,

- For GPT-3.5/GPT-4, an OpenAI API key is needed. Replace the placeholder with your key in `src/config.py`.

- Git-lfs is required to download and load corpora for the first time.

- Java is requried for using BM25.

## Usage

Currently, we provide supports for single corpora and retrievers.

```python
from src.medrag import MedRAG

question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
options = {
    "A": "paralysis of the facial muscles.",
    "B": "paralysis of the facial muscles and loss of taste.",
    "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
    "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
}

## CoT Prompting
cot = MedRAG(llm_name="OpenAI/gpt-3.5-turbo-16k", rag=False)
answer, _, _ = cot.answer(question=question, options=options)

## MedRAG
medrag = MedRAG(llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, retriever_name="MedCPT", corpus_name="Textbooks")
answer, snippets, scores = medrag.answer(question=question, options=options, k=32) # scores are given by the retrieval system
```

## Citation
```
@article{xiong2024benchmarking,
    title={Benchmarking Retrieval-Augmented Generation for Medicine}, 
    author={Guangzhi Xiong and Qiao Jin and Zhiyong Lu and Aidong Zhang},
    journal={arXiv preprint arXiv:2402.13178},
    year={2024}
}
```