#!/usr/bin/env python3
"""
rag_chroma.py

Refactored from: RAG_Vector_store_chroma.ipynb
Source notebook: https://github.com/RobelDawit/RAG-applications/blob/a32a108d87327a56e778e76adcee8dfa1efaedcd/RAG_Vector_store_chroma.ipynb

Simple CLI to build or query a Chroma vector store from a PDF using OpenAI embeddings.
"""

import os
import argparse
import logging
import textwrap
from typing import List, Optional

# LangChain / vectorstore imports
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_env_from_prompt(var_name: str, force: bool = False) -> None:
    """
    Prompt user to enter a value for an environment variable if not set.
    Useful in interactive runs (like Colab).
    """
    if force or not os.environ.get(var_name):
        try:
            # getpass hides input; prefer plain input if not available
            import getpass

            value = getpass.getpass(f"{var_name}: ")
        except Exception:
            value = input(f"{var_name}: ")
        os.environ[var_name] = value


def load_pdf_documents(pdf_path: str) -> List:
    """
    Load a PDF into LangChain Document objects using PyPDFLoader.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    logger.info("Loading PDF: %s", pdf_path)
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    logger.info("Loaded %d raw documents/pages", len(docs))
    return docs


def split_documents(documents: List, chunk_size: int = 500, chunk_overlap: int = 50) -> List:
    """
    Split documents into smaller chunks for embedding.
    """
    logger.info("Splitting documents with chunk_size=%d overlap=%d", chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(documents)
    logger.info("Split into %d chunks", len(split_docs))
    return split_docs


def create_embeddings(openai_api_key: Optional[str] = None):
    """
    Create an embeddings object. If openai_api_key provided, pass explicitly.
    """
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Call set_env_from_prompt or set the env var.")
    logger.info("Creating OpenAI embeddings client")
    return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


def build_chroma_vector_store(documents, embeddings, persist_directory: str):
    """
    Build and persist a Chroma vector store from documents + embeddings.
    """
    logger.info("Creating Chroma vector store at %s", persist_directory)
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    # If using a local Chroma server, consider different init path
    logger.info("Persisted vector store to %s", persist_directory)
    return vectordb


def load_chroma_vector_store(embeddings, persist_directory: str):
    """
    Load an existing Chroma vector store (persisted).
    """
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Persist directory not found: {persist_directory}")
    logger.info("Loading Chroma vector store from %s", persist_directory)
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def run_query_on_store(vector_store, query: str, k: int = 10) -> str:
    """
    Run a RetrievalQA chain using OpenAI LLM and the provided vector store retriever.
    Returns the answer string.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    llm = OpenAI()  # If you wish to pass model name, temperature, do so here: OpenAI(model_name="gpt-4o", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    logger.info("Running query against vector store (k=%d)", k)
    response = qa_chain.run(query)
    return response


def main():
    parser = argparse.ArgumentParser(description="Build / query a Chroma vector store from a PDF using OpenAI embeddings.")
    parser.add_argument("--pdf", "-p", type=str, help="Path to PDF file to index")
    parser.add_argument("--persist-dir", "-d", type=str, default="./chroma_db", help="Directory to persist Chroma DB")
    parser.add_argument("--create", action="store_true", help="Create/persist the vector store from --pdf")
    parser.add_argument("--query", "-q", type=str, help="Run a query against the persisted vector store")
    parser.add_argument("--k", type=int, default=10, help="Number of docs to retrieve (k)")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--openai-api-key", type=str, default=None, help="OpenAI API Key (optional, can use OPENAI_API_KEY env var)")
    args = parser.parse_args()

    # Ensure API key available
    if not args.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found in env; prompting interactively.")
        set_env_from_prompt("OPENAI_API_KEY")

    embeddings = create_embeddings(args.openai_api_key)

    if args.create:
        if not args.pdf:
            parser.error("--create requires --pdf PATH")
        docs = load_pdf_documents(args.pdf)
        split_docs = split_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        vectordb = build_chroma_vector_store(split_docs, embeddings, persist_directory=args.persist_dir)
        logger.info("Vector store built and persisted.")
        # Optional: return or print stats
    if args.query:
        vectordb = load_chroma_vector_store(embeddings, persist_directory=args.persist_dir)
        answer = run_query_on_store(vectordb, args.query, k=args.k)
        print("\nAnswer:\n")
        print(textwrap.fill(answer, width=100))


if __name__ == "__main__":
    main()