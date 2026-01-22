# Create a simple script: find_pages.py
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if pinecone_api_key:
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name="medicalbot"
)

questions = [
    "What is acne?",
    "What are common symptoms of asthma?",
    "What is anemia?",
    "What is appendicitis?",
    "What causes Alzheimer's disease?",
    "What is diabetes mellitus?",
    "What is hypertension?",
    "What are common risk factors for breast cancer?",
    "What is Parkinson disease?",
    "What is osteoarthritis?",
    "What is pneumonia?",
    "What is bipolar disorder?",
    "What is a cataract?",
    "What are symptoms of dehydration?",
    "What is eczema?",
    "What is fibromyalgia?",
    "What is gout?",
    "What is hepatitis?",
    "What is influenza?",
    "What is a kidney stone?"
]

for q in questions:
    docs = docsearch.similarity_search(q, k=3)
    pages = set()
    for doc in docs:
        page = doc.metadata.get("page_display") or doc.metadata.get("page")
        if page:
            pages.add(int(page))
    print(f"{q}: {sorted(pages)}")