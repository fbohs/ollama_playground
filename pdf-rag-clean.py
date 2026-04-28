# main.py

import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = "./data/BOI.pdf"
MODEL_NAME = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"


def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


def create_vector_db(chunks):
    """Create a vector database from document chunks."""
    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING_MODEL)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=VECTOR_STORE_NAME,
    )
    logging.info("Vector database created.")
    return vector_db


def create_retriever(vector_db, llm):
    """Create a standard retriever."""
    retriever = vector_db.as_retriever()
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain"""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created successfully.")
    return chain


def main():
    # Load and process the PDF document
    data = ingest_pdf(DOC_PATH)
    if data is None:
        return

    # Split the documents into chunks
    chunks = split_documents(data)

    # Create the vector database
    vector_db = create_vector_db(chunks)

    # Initialize the language model
    llm = ChatOllama(model=MODEL_NAME)

    # Create the retriever
    retriever = create_retriever(vector_db, llm)

    # Create the chain with preserved syntax
    chain = create_chain(retriever, llm)

    # Example query
    question = "How to report BOI?"

    # Get the response
    res = chain.invoke(input=question)
    print("Response:")
    print(res)


if __name__ == "__main__":
    main()
