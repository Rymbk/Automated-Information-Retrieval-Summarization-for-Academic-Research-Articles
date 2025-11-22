import os
import glob
import fitz
from dotenv import load_dotenv
import chromadb
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from evaluate import load
import matplotlib.pyplot as plt
import gradio as gr
from smolagents import OpenAIServerModel, CodeAgent, ToolCallingAgent, HfApiModel, GradioUI ,tool
from smolagents.tools import Tool
import requests
from typing import Dict, List, Union,Tuple
import re
import torch
import textwrap
from tqdm import tqdm
import gc


load_dotenv()

reasoning_model_id = os.getenv("REASONING_MODEL_ID")
tool_model_id = os.getenv("TOOL_MODEL_ID")
huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
db_dir = r"C:\Users\ACER NITRO\OneDrive\Bureau\Project 2SCI\VectorDB_Embeddings"
data_dir = "./data"
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)
# Device Config
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

def get_model(model_id):
    using_huggingface = os.getenv("USE_HUGGINGFACE", "yes").lower() == "yes"
    if using_huggingface:
        return HfApiModel(model_id=model_id, token=huggingface_api_token)
    else:
        return OpenAIServerModel(
            model_id=model_id,
            api_base="http://localhost:11434/v1",
            api_key="ollama"
        )
    

def ollama_chat(prompt, model=reasoning_model_id):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False  
        }
    )
    response.raise_for_status()
    result = response.json()
    return result['response']
#outputs collapsed

# Initialize models
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
reasoning_model = get_model(reasoning_model_id)
tool_model = get_model(tool_model_id)
reasoner = get_model(reasoning_model_id)

# Initialize ChromaDB
client = chromadb.PersistentClient(path=db_dir)
collection = client.get_or_create_collection(name='ties_collection_emb', metadata={"hnsw:space": "cosine"})


def process_scientific_text(text):
    # Remove references (e.g., [10], [15])
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove formulas (e.g., Lntp = ..., Lclass = ...)
    text = re.sub(r'L[a-zA-Z]+\s*=\s*[^=]+', '', text)
    
    # Remove table-related content (e.g., "TABLE 1. OVERVIEW OF THE DGA DATASET")
    text = re.sub(r'TABLE \d+\..*?\n', '', text, flags=re.IGNORECASE)
    
    # Remove numerical results (e.g., 97%, 0.7%, 1458863)
    text = re.sub(r'\b\d+%|\b\d+\.\d+%|\b\d{3,}\b', '', text)
    
    # Remove dataset-specific details (e.g., URLs, dataset sizes)
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'\b\d+\s*(domains|records|samples|queries)\b', '', text, flags=re.IGNORECASE)
    
    # Remove lines with DGA dataset examples (e.g., zsvubwnqlefqv.com, xshellghost)
    text = re.sub(r'^\w+\s+\d+\s+[^\s]+\.[a-z]+$', '', text, flags=re.MULTILINE)
    
    # Remove empty lines and normalize whitespace, preserving section headings
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = '\n'.join(lines)  # Preserve line breaks for section headings
    
    return text

# PDF processing
def extract_and_chunk_pdf(file_path, chunk_size=800, chunk_overlap=400):
    """Extracts text from a PDF and splits into chunks."""
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text("text") for page in doc])
    
    # Extract abstract (assuming it's the first paragraph or labeled)
    abstract = text.split("\n\n")[0] if "abstract" in text.lower() else ""
    
    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    chunks = [process_scientific_text(chunk) for chunk in chunks]
    return chunks, abstract

def compute_embeddings(chunks):
    """ Computes embeddings for text chunks """
    return embedding_model.encode(chunks, convert_to_numpy=True)

def store_in_vector_db(chunks, file_path):
    """Stores chunks and embeddings in ChromaDB."""
    doc_id = os.path.basename(file_path)
    embeddings = compute_embeddings(chunks)
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[f"{doc_id}_chunk_{i}"],
            documents=[chunk],
            embeddings=[embedding.tolist()],
            metadatas=[{"source": doc_id, "chunk_id": i}]
        )
    
    return len(chunks)

def extract_sections(paper_text: str) -> str:
        """
        Extracts key sections from a scientific paper using regex.

        Args:
            paper_text (str): The full text of the scientific paper.

        Returns:
            str: A string containing the extracted sections formatted as "Section: Text".
        """
        sections = {}
        section_patterns = {
            "Abstract": r"(?i)^Abstract\n([\s\S]*?)(?=\n\n(?:Introduction|Methods|Results|Discussion|Conclusion|\Z))",
            "Introduction": r"(?i)^Introduction\n([\s\S]*?)(?=\n\n(?:Methods|Results|Discussion|Conclusion|\Z))",
            "Methods": r"(?i)^Methods\n([\s\S]*?)(?=\n\n(?:Results|Discussion|Conclusion|\Z))",
            "Results": r"(?i)^Results\n([\s\S]*?)(?=\n\n(?:Discussion|Conclusion|\Z))",
            "Discussion": r"(?i)^Discussion\n([\s\S]*?)(?=\n\n(?:Conclusion|\Z))",
            "Conclusion": r"(?i)^Conclusion\n([\s\S]*?)(?=\n\n|\Z)"
        }
        
        for section, pattern in section_patterns.items():
            match = re.search(pattern, paper_text, re.MULTILINE)
            if match:
                sections[section] = match.group(1).strip()
        
        if sections:
            sections_str = "\n\n".join([f"{key}:\n{value}" for key, value in sections.items()])
        else:
            sections_str = "Full Text:\n" + paper_text
        
        return sections_str

def retrieve_similar_chunks(embeddings: np.ndarray, sections: str, top_k: int = 5) -> List[str]:
        """Retrieve top-k similar chunks from ChromaDB by matching each chunk separately."""
    
        client = chromadb.PersistentClient(path=db_dir)
        collection = client.get_or_create_collection(name='ties_collection_emb', metadata={"hnsw:space": "cosine"})
        doc_embedding = np.mean(embeddings, axis=0).tolist()
        results = collection.query(query_embeddings=[doc_embedding], n_results=top_k)
        chunks = results["documents"][0] if results["documents"] else []
    
        # Prioritize chunks from Abstract and Conclusion
        prioritized = []
        sections_lower = sections.lower()
        for chunk in chunks:
            if any(section in chunk.lower() for section in ["abstract", "conclusion"]) or any(section in sections_lower for section in ["abstract", "conclusion"]):
                prioritized.insert(0, chunk)
            else:
                prioritized.append(chunk)

        return prioritized[:top_k]

def baseline_model(pdf_path , query):
    """Baseline: Summarizes PDF chunks without RAG."""
    pdf_chunks = extract_and_chunk_pdf(pdf_path)[0]
    context = "\n".join(pdf_chunks) 
    prompt = f"""
    {query} this Scientific Paper:
    {context}
    """
    response = ollama_chat(prompt, model=reasoning_model_id)
    clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return clean_response

def classical_rag(pdf_path, user_query):
    """Classical RAG: Retrieves chunks based on query and summarizes."""
    pdf_chunks = extract_and_chunk_pdf(pdf_path)[0]
    sections = extract_sections("\n".join(pdf_chunks))
    results = retrieve_similar_chunks(compute_embeddings(pdf_chunks),sections, top_k=5)
    pdf_chunks = "\n".join(pdf_chunks[1::9])
    context = "\n".join(results) 
    prompt = f"""
You are a helpful and intelligent AI assistant specialized in summarizing scientific papers. Your task is to generate a comprehensive summary based on the provided PDF chunks, while enriching your response with relevant insights drawn from the retrieved contextual information.

Use the **PDF Context** as your primary source for the paper's content, and consult the **Retrieved Context** to enhance understanding, clarify technical terms, and provide broader perspective when necessary.

User Query:
{user_query}

PDF Context (extracted from the paper):
{pdf_chunks}

Retrieved Context (external relevant information):
{context}

Please ensure your summary directly addresses the user's query, remains grounded in the content of the PDF, and is enhanced—but not contradicted—by the retrieved context.
"""
    response = ollama_chat(prompt, model=reasoning_model_id)
    clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return clean_response

@tool
def rag_with_reasoner(user_query: str, pdf_path: str) -> str:
    """
    Agentic RAG: Uses PDF chunks and retrieved context with reasoning.

    Args:
        user_query: The user query string.
        pdf_path: A list of text chunks from the PDF.
        pdf_embeddings: A numpy array containing the embeddings for the PDF chunks.

    Returns:
        A summary or answer generated by the reasoning agent.
    """
    # Create the reasoner for better RAG
    reasoning_model = get_model(reasoning_model_id)
    reasoner = CodeAgent(tools=[], model=reasoning_model, add_base_tools=False, max_steps=2)
    pdf_chunks= extract_and_chunk_pdf(pdf_path)[0]
    pdf_embeddings = compute_embeddings(pdf_chunks)
    sections = extract_sections("\n".join(pdf_chunks))
    retrieved_chunks = retrieve_similar_chunks(pdf_embeddings , sections , top_k=5)
    pdf_context = "\n".join(pdf_chunks[1::9]) # to avoid too long context
 
    prompt = f"""
    You are a scientific paper summarizer. Be concise and specific to help scientific community.
    Prioritize the uploaded document's content and use additional context only for enrichment.

    Uploaded Document:
    {pdf_context}

    Additional Context:
     f"Key Sections:\n{sections}\n\nPrioritized Chunks:\n{'\n'.join(retrieved_chunks)} 

    Query: {user_query}

     Answer:
     """
    response = reasoner.run(prompt, reset=False).split("</think>")[-1].strip()
    clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    return clean_response

