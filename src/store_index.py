from src.helper import load_pdf_data, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_data("../Data/")
text_chunks = text_split(extracted_data)
print("Chunks:", len(text_chunks))

embeddings = download_hugging_face_embeddings()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "medicalbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
        )
    )

# we embed each chunk and upsert the embeddings into our Pinecone Index
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings, 
    index_name=index_name
)

print("✅ Indexing complete.")






