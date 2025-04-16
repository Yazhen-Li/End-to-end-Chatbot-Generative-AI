from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os

def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)

    documents = loader.load()

    return documents

def text_split(extracted_data):
    text_split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_split.split_documents(extracted_data)

    return text_chunks

def download_hugging_face_embeddings():
    # this model return 384 dimensions
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    return embeddings

def clear_data():
    os.system("rm -rf data/*")
    os.system("rm -rf db/*")
