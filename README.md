## Paperly
## Automated Information Retrieval and Summarization for Academic Research Articles

## Description

This repository contains the implementation of an OCR + LLM-powered tool designed to assist researchers, students, and academics in extracting, summarizing, and organizing information from academic research papers, including PDFs and scanned documents. The project focuses on a comparative study of retrieval-augmented summarization techniques, evaluating approaches such as Baseline, RAG, AgenticRAG, and GraphRAG using models like DeepSeek1.5, BART-large-cnn-samsum, T5-Small, and Mistral.

The tool addresses the challenges of processing complex scientific literature by integrating information retrieval with language generation, ensuring contextually accurate and domain-specific summaries.

## Problem Statement

Scientific papers are complex and voluminous, making manual summarization labor-intensive. Automated systems often fail to capture nuanced contexts or handle specialized terminology effectively. Traditional language models lack external knowledge integration, resulting in generic summaries. This project aims to: 

1. Evaluate RAG, AgenticRAG, and GraphRAG with DeepSeek1.5, T5-Small, BART-large-cnn-samsum, and Mistral for scientific paper summarization.
2. Identify optimal chunking strategies for retrieval and summarization quality.
3. Determine the most efficient approach for deployment.

## Introduction

The rapid expansion of scientific literature has created a pressing need for intelligent systems capable of extracting meaningful insights from research articles. Large Language Models (LLMs), such as DeepSeek1.5, are powerful in generating fluent text but often fall short in factual grounding and domain-specific summarization. Retrieval-Augmented Generation (RAG) mitigates these limitations by incorporating relevant external information from a VectorDB (VectorDB), thereby improving contextual accuracy.

Experiments were conducted using the DeepSeek1.5 model across all approaches. In addition, Classical RAG and AgenticRAG were also tested with Mistral, T5-Small, and google/flan-bart-large-cnn-samsum to evaluate generalization across LLM backbones. Due to computational constraints, GraphRAG was implemented exclusively with DeepSeek1.5 and limited to a subset of 15 papers.

Additionally, RAG and AgenticRAG were tested with Mistral, T5-Small, and BART-large-cnn-samsum models. Evaluation used 14,000+ papers and four representative queries. GraphRAG with 800/400 chunking yielded the best metric scores and execution time.

## System Overview

This project evaluates four summarization strategies across multiple models:

| Approach     | Models Tested                          | Key Mechanism                          | Constraints                     |
|--------------|----------------------------------------|----------------------------------------|---------------------------------|
| Baseline    | DeepSeek1.5, T5-Small, BART-large-cnn-samsum, Mistral | Direct summarization without retrieval | Limited by context window      |
| RAG         | DeepSeek1.5, T5-Small, BART-large-cnn-samsum, Mistral | VectorDB retrieval with embeddings     | Retrieval accuracy depends on chunking |
| AgenticRAG  | DeepSeek1.5, T5-Small, BART-large-cnn-samsum, Mistral | Retrieval with iterative reasoning     | Higher computational cost       |
| GraphRAG    | DeepSeek1.5                           | Knowledge graph-based retrieval        | Limited to 15 papers due to resources |

## Dataset

We constructed a VectorDB containing over 14,000 scientific papers sourced from repositories like arXiv and PubMed, covering domains such as computer science, biomedical sciences, and physics. The papers were preprocessed to remove non-contextual elements (e.g., tables, formulas) using a custom function. The VectorDB was built with three chunking configurations:

| Chunk Size | Overlap     | Use Case                              |
|------------|-------------|---------------------------------------|
| 500 tokens| 200 tokens | High granularity for dense sections (e.g., methods) |
| 800 tokens| 400 tokens | Balanced context for general scientific texts |
| 1200 tokens| 300 tokens | Narrative sections with reduced computational load |

Text embeddings were generated using all-MiniLM-L6-v2 (8192-token context) for compatibility with larger chunks, stored in a VectorDB using FAISS for efficient similarity search. For GraphRAG, we constructed a knowledge graph from 15 papers due to computational and time constraints, using Microsoft’s GraphRAG framework with DeepSeek1.5. The graph modeled entities (e.g., methods, concepts) and relationships, enhancing retrieval for complex queries. The dataset diversity ensured robust testing across domains, with chunking variations allowing us to evaluate retrieval and summarization performance comprehensively.

## Preprocessing Strategy

To ensure high-quality summarization, we developed a preprocessing function to clean scientific paper text, removing non-contextual elements while retaining narrative content and section headings. The function performs the following steps:

- Removes references (e.g., [10]).
- Eliminates formulas (e.g., Lclass = ...).
- Excludes tables (e.g., "TABLE 1. OVERVIEW...").
- Filters out numerical results (e.g., 97%, 1458863).
- Removes URLs and dataset-specific details (e.g., domain counts).
- Eliminates DGA dataset examples (e.g., zsvubwnqlefqv.com).

This preprocessing ensured clean input for the VectorDB and GraphRAG, improving retrieval accuracy and summary coherence.

## Approaches and Results

The project implements and compares the following approaches:

- **Baseline**: Direct summarization without external retrieval.
- **RAG**: Uses VectorDB for context augmentation via similarity-based retrieval.
- **AgenticRAG**: Extends RAG with a reasoning agent for iterative context refinement.
- **GraphRAG**: Leverages a knowledge graph to model entity relationships for enhanced retrieval.

Evaluations were conducted using metrics such as ROUGE-1, ROUGE-L, BLEU, and BERTScore across different chunk sizes (500/200, 800/400, 1200/300). GraphRAG with 800/400 chunking achieved the highest scores (ROUGE-1: 0.45, ROUGE-L: 0.40, BLEU: 0.20, BERTScore: 0.85) but at a higher computational cost.

## Deployment

Given GraphRAG’s superior performance with chunk size 800 and overlap 400, it stood out as the most accurate summarization model. However, to balance performance with responsiveness, we opted to deploy AgenticRAG, which delivered results close to GraphRAG’s in terms of quality while maintaining a substantially more reasonable execution time.

The deployment was carried out on a cloud-based infrastructure, integrating Neo4j for knowledge graph storage and DeepSeek1.5 hosted on a GPU cluster to ensure scalability across large scientific corpora. The system supports real-time query processing, delivering concise, query-specific summaries to researchers.

Access is provided through a web interface, allowing users to submit queries and receive summaries instantly. Scalability tests confirmed the system can handle up to 100 concurrent queries efficiently.

## MLOps

This section details the MLOps strategy for our Retrieval Augmented Generation (RAG) system, which utilizes the best model for scientific paper summarization. The core objective is to leverage MLflow to manage experiments, track model performance, and facilitate dynamic updates to system components such as prompts and embedding models. This MLOps framework aims to enable iterative improvements and robust evaluation.

Key MLOps Practices Implemented with MLflow:

- **Metric Logging per Query**: For every submitted query and chosen configuration, the system logs detailed performance metrics (ROUGE, BERTScore, execution time) for each summarization method (Baseline, RAG, AgenticRAG) directly to MLflow.
- **Dynamic Embedding Model Management**: The system allows selection of different embedding models via the interface. Each selection is treated as a distinct experimental parameter and logged to MLflow.
- **Interactive Prompt Engineering and Versioning**: The system allows modification of prompt templates for the RAG and AgenticRAG approaches through the UI. Every modified prompt used in a summarization run is logged as an artifact to MLflow.

## Future Work

Future enhancements, prioritized by impact, include:

- Multi-Modal Summarization: Integrate tables and figures using OCR and image-to-text models to capture comprehensive paper content.
- Reference Extraction: Automatically summarize cited works for deeper context.
- Dynamic Chunking: Use semantic chunking for adaptive chunk sizes.
- Fine-Tuning: Enhance model performance on domain-specific terminology.
- Scalability: Optimize GraphRAG for larger corpora using distributed computing.



## Included Images

### Web Interface
![Web Interface](https://github.com/ADHAYA-Technos/Automated-Information-Retrieval-and-Summarization-for-Academic-Research-Articles/blob/main/web_interface_s.png)
- **Description**: This screenshot showcases the web interface of the deployed summarization tool, allowing users to submit queries and receive real-time summaries from academic papers.

### Preprocessing Workflow
![Preprocessing Workflow](https://github.com/ADHAYA-Technos/Automated-Information-Retrieval-and-Summarization-for-Academic-Research-Articles/blob/main/preprocessing_workflow.png)
- **Description**: This architecture diagram illustrates the preprocessing workflow, detailing steps like text extraction, chunking, and cleaning to prepare data for the VectorDB and GraphRAG.

### GraphRAG Example
![GraphRAG Example](https://github.com/ADHAYA-Technos/Automated-Information-Retrieval-and-Summarization-for-Academic-Research-Articles/blob/main/graphrag.png)
- **Description**: This image depicts the first layer of a knowledge graph generated by GraphRAG, showcasing entity relationships (e.g., methods, concepts) from a subset of 15 papers using DeepSeek1.5.

### Metric Scores Comparison for DeepSeek1.5
![Metric Scores Comparison](https://github.com/ADHAYA-Technos/Automated-Information-Retrieval-and-Summarization-for-Academic-Research-Articles/blob/main/Deepseek-queries_comparisions.jpg)
- **Description**: This chart compares average metric scores (ROUGE-1, ROUGE-L, BERTScore) for queries like "What problem does this paper address and why is it important?", "Explain the main methodology used in this paper", "Summarize this paper in 200 words?", and "What are the main results and contributions of this paper?" across Baseline(DeepSeek1.5), Classical-RAG, AgenticRAG, and GraphRAG.
## Conclusion

This project rigorously compared RAG, AgenticRAG, and GraphRAG across DeepSeek1.5, T5-Small, BART-large-cnn-samsum, and Mistral for scientific paper summarization. GraphRAG with 800/400 chunking achieved superior performance (ROUGE-1: 0.45, BERTScore: 0.85), leveraging knowledge graphs to enhance contextual accuracy. The deployed system offers a scalable, query-focused solution for researchers, validated on a diverse corpus of 14,000 papers. This work advances automated literature analysis, setting a foundation for intelligent, multi-modal scientific discovery tools.

## Contributors
- Abdelillah SERGHINE
- Lamia ABDELMALEK
- Mariam BOUKENNOUCHE
- Souad BOUKHAROUBA
- Yacine DAIT DEHANE

**Supervisor**: Mr. Khaldi B.

 
For detailed member contributions, refer to the project report or the GitHub link in the documentation.

 
