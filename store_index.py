from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma

from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings

load_dotenv()

embeddings = download_hugging_face_embeddings()

documents = load_pdf_file('./data')
text_chunks = text_split(documents)

vectordb = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory='./db')
vectordb.persist()
