from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name="medicalbot"
)

# Questions without expected_pages
questions = [
    "What is acne?",
    "What is anemia?",
    "What is appendicitis?",
    "What causes Alzheimer's disease?",
    "What is hypertension?",
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
    docs = docsearch.similarity_search(q, k=10)
    pages = sorted(set([d.metadata.get("page_display") for d in docs if d.metadata.get("page_display")]))
    print(f'"{q}": {pages}')