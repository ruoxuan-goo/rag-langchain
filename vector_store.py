## ---Build the vectore store for the given data set ---###
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, PDFMinerLoader
import os
import uuid
from pinecone import Pinecone, ServerlessSpec
import chainlit as cl

#Global Variables
chunk_size = 1024
chunk_overlap = 100
PDF_STORAGE_PATH = "./pdfs"

# Load the environment variables
load_dotenv()

# Pinecone client
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "rag-langchain"

# Initialize the Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, metric="cosine", dimension=1536, spec=ServerlessSpec(cloud="aws", region="us-east-1"))

# Specify the embeddings model
embeddings = OpenAIEmbeddings()

def process_pdf(pdf_storage_path: str):
    pdf_directory = Path(pdf_storage_path)
    docs = []
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Load pdfs and split into documents
    for pdf_path in pdf_directory.glob("*.pdf"): 
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()
        docs += text_spliter.split_documents(documents)
        
    # Convert text to embeddings
    for doc in docs:
        embedding = embeddings.embed_query(doc.page_content)
        random_id = str(uuid.uuid4())
        # print (embedding)
        doc_search = pc.Index(index_name)
        # Store the vector in Pinecone index
        doc_search.upsert(
            vectors=[
                {
                    "id": random_id,
                    "values": embedding,
                    "metadata": {
                        "source": doc.page_content,
                    }
                }
            ],
            namespace="ai-agent-survey"
        )
        print(f"Document {random_id} stored in Pinecone")
    return doc_search

def main():
    process_pdf(PDF_STORAGE_PATH)

if __name__ == "__main__":
    main()
        