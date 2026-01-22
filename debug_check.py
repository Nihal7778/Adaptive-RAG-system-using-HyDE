from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name="medicalbot"
)

# Check what doc_ids exist in Pinecone
# Try a simple search without filters
results = docsearch.similarity_search("test", k=10)

print("\n=== CHECKING INDEXED DOCUMENTS ===")
doc_ids = set()
for r in results:
    doc_id = r.metadata.get("doc_id", "NO_DOC_ID")
    doc_name = r.metadata.get("doc_name", "NO_NAME")
    doc_ids.add(doc_id)
    print(f"doc_id: {doc_id} | doc_name: {doc_name}")

print(f"\n=== UNIQUE DOC_IDS: {doc_ids} ===")