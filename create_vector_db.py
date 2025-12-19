import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration
API_KEY = "AIzaSyAx9TunkefuJeNoCuex4p7xTh2akH_VOSU"
os.environ["GOOGLE_API_KEY"] = API_KEY

KNOWLEDGE_FILE = "knowledge.txt"
FAISS_INDEX_PATH = "faiss_index"

def create_vector_db():
    if not os.path.exists(KNOWLEDGE_FILE):
        print(f"Error: {KNOWLEDGE_FILE} not found.")
        return

    print("Loading knowledge base...")
    loader = TextLoader(KNOWLEDGE_FILE, encoding="utf-8")
    documents = loader.load()

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks.")

    print("Generating embeddings and creating vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_db = FAISS.from_documents(texts, embeddings)

    print(f"Saving index to {FAISS_INDEX_PATH}...")
    vector_db.save_local(FAISS_INDEX_PATH)
    print("Success! Vector database created.")

if __name__ == "__main__":
    create_vector_db()
