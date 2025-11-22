# Multi-Agent RAG for Scientific Paper Summarization

This project implements a **Multi-Agent Retrieval-Augmented Generation (RAG)** pipeline for summarizing scientific papers in PDF format. The workflow processes PDFs, extracts key sections, generates embeddings, retrieves relevant chunks, and produces a concise 200-word summary using a combination of static functions and agent-based guidance. The system leverages large language models (LLMs) and vector databases to ensure accurate and contextually relevant summaries.

## Features

- **Static Processing**: Deterministic functions for extracting sections, chunking text, computing embeddings, and retrieving relevant content.
- **Agent-Based Guidance**: LLMs guide the workflow by refining and validating summaries for clarity and accuracy.
- **Vector Database**: Uses ChromaDB with `sentence-transformers/all-MiniLM-L6-v2` embeddings for efficient retrieval.
- **Evaluation**: Computes ROUGE, BLEU, and BERTScore metrics to assess summary quality.
- **Gradio Interface**: Provides a user-friendly web interface for uploading PDFs and querying summaries.

## Installation

### Prerequisites

- Python 3.8+
- Git
- A HuggingFace API token (for model access)
- A PDF file for testing (e.g., `A Framework for Fine-Tuning LLMs using Heterogeneo.pdf`)

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/multi-agent-rag-summarization.git
   cd multi-agent-rag-summarization
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   Create a `.env` file in the project root and add:
   ```env
   REASONING_MODEL_ID=your-reasoning-model-id
   TOOL_MODEL_ID=your-tool-model-id
   HUGGINGFACE_API_TOKEN=your-huggingface-token
   USE_HUGGINGFACE=yes
   ```
   Replace placeholders with your model IDs (e.g., `qwen2.5:3b-instruct`) and HuggingFace token.

5. **Download Sample Data**:
   Place PDF files in the `./data` directory. A sample PDF (`A Framework for Fine-Tuning LLMs using Heterogeneo.pdf`) can be used for testing.

## Usage

### Running the Script

1. **Summarize a Single PDF**:
   ```bash
   python multi_agent_rag_workflow_static.py
   ```
   This runs the script with the default PDF (`./data/A Framework for Fine-Tuning LLMs using Heterogeneo.pdf`) and outputs a 200-word summary.

2. **Batch Evaluation**:
   The script evaluates all PDFs in the `./data` directory and generates plots for ROUGE, BLEU, and BERTScore metrics, saved in the `./results` directory.

3. **Gradio Interface**:
   Launch the web interface to upload PDFs and query summaries:
   ```bash
   python multi_agent_rag_workflow_static.py
   ```
   Access the interface at `http://127.0.0.1:7860` in your browser.

### Example Output

Running the script with the sample PDF produces a summary like:
```
Summary: The paper presents a framework for fine-tuning large language models (LLMs) using heterogeneous feedback, addressing challenges in dataset collection and supervision format variability. It combines diverse feedback into a unified format compatible with supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF), enhancing model performance across tasks like text summarization and chatbots. The framework's two main components include feedback unification and iterative model tuning, achieving improved accuracy and robustness. Extensive experiments demonstrate its effectiveness in handling numerical, binary, and multi-dimensional feedback, making it a versatile solution for LLM optimization. The authors, from Carnegie Mellon University and Adobe Research, highlight its scalability and potential for real-world applications.
```

## File Structure

```
multi-agent-rag-summarization/
├── data/                           # Directory for input PDF files
├── results/                        # Directory for evaluation plots
├── assets/                         # Directory for static assets (e.g., paperly.png)
├── .env                            # Environment variables (not tracked)
├── requirements.txt                # Python dependencies
├── multi_agent_rag_workflow_static.py  # Main script
├── README.md                       # This file
```

## Dependencies

Key libraries include:
- `fitz` (PyMuPDF): PDF text extraction
- `sentence-transformers`: Embedding generation
- `chromadb`: Vector database
- `langchain`: Text chunking
- `evaluate`: Metric computation
- `gradio`: Web interface
- `smolagents`: Agent-based orchestration

See `requirements.txt` for the full list.

## Workflow Overview

The pipeline consists of the following steps:

1. **Input Processing**: Extracts sections (e.g., Abstract, Conclusion) and chunks text using static functions.
2. **Embedding and Retrieval**: Generates embeddings with `all-MiniLM-L6-v2`, stores them in ChromaDB, and retrieves relevant chunks.
3. **Summary Generation**(maybe still not used in the actual version): Produces a draft summary using a reasoning agent.
4. **Fact-Checking**(maybe still not used in the actual version): Validates the summary against the original text.
5. **Refinement and Validation**(maybe still not used in the actual version): Agents refine the summary for clarity and validate factual accuracy.
6. **Evaluation**: Computes ROUGE, BLEU, and BERTScore metrics.

Agents (`reasoning_agent`, `tool_agent`, `coordinator_agent`) guide the process by refining and validating outputs without relying on tool-calling.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please include tests and update documentation as needed.

## Issues

If you encounter issues (e.g., model errors, dependency conflicts), check the following:

- Ensure the `.env` file is correctly configured.
- Verify the `smolagents` library version (`pip show smolagents`).
- Confirm PDF files are in the `./data` directory.

Report issues on the [GitHub Issues](https://github.com/ADHAYA-Technos/Automated-Information-Retrieval-and-Summarization-for-Academic-Research-Articles/issues) page with detailed logs.

## License


## Acknowledgments


- Uses models from HuggingFace and the `sentence-transformers` library.
- Inspired by research on RAG and multi-agent systems for NLP.

---

**Author**: [DAIT DEHANE Yacine]  
**Contact**: [y.daitdehane@esi-sba.dz]  
**Last Updated**: April 28, 2025