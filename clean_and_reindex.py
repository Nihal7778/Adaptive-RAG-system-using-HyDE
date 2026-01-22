"""
Clean slate: Delete everything and re-index from scratch
Run this ONCE to fix the NO_DOC_ID issue
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_data, text_split, download_hugging_face_embeddings
from src.document_manager import update_document_status

load_dotenv()

print("🧹 CLEAN SLATE RE-INDEXING")
print("="*50)

# 1. Load PDFs
print("\n📄 Step 1: Loading PDFs from Data/...")
extracted_data = load_pdf_data("Data/")
print(f"   ✅ Loaded {len(extracted_data)} document pages")

# 2. Split into chunks with doc_id
print("\n✂️ Step 2: Splitting into chunks...")
text_chunks = text_split(extracted_data)
print(f"   ✅ Created {len(text_chunks)} chunks")

# 3. Verify all chunks have doc_id
print("\n🔍 Step 3: Verifying doc_id metadata...")
doc_ids_found = set()
no_doc_id_count = 0
for chunk in text_chunks:
    doc_id = chunk.metadata.get("doc_id")
    if doc_id:
        doc_ids_found.add(doc_id)
    else:
        no_doc_id_count += 1

print(f"   ✅ Found doc_ids: {doc_ids_found}")
if no_doc_id_count > 0:
    print(f"   ⚠️ WARNING: {no_doc_id_count} chunks have NO doc_id!")
    print("   This means metadata.json is missing entries.")
    exit(1)

# 4. Connect to Pinecone
print("\n🔌 Step 4: Connecting to Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "medicalbot"
index = pc.Index(index_name)

# 5. Delete ALL existing vectors
print("\n🗑️ Step 5: Deleting ALL existing vectors...")
try:
    # Get stats
    stats = index.describe_index_stats()
    total_vectors = stats.get('total_vector_count', 0)
    print(f"   Current vectors in index: {total_vectors}")
    
    if total_vectors > 0:
        # Delete by namespace (if using default namespace)
        print("   Deleting all vectors...")
        # For serverless, we use delete with deleteAll=True
        index.delete(delete_all=True)
        print("   ✅ All vectors deleted")
    else:
        print("   ℹ️ Index already empty")
        
except Exception as e:
    print(f"   ⚠️ Delete failed: {e}")
    print("   Trying alternative method...")
    # Alternative: delete by dummy filter
    try:
        index.delete(filter={})
    except:
        pass

# 6. Load embeddings
print("\n🧠 Step 6: Loading embeddings model...")
embeddings = download_hugging_face_embeddings()

# 7. Upload new vectors
print("\n⬆️ Step 7: Uploading new vectors...")
print(f"   This will take ~2-3 minutes for {len(text_chunks)} chunks...")

PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print("   ✅ Upload complete!")

# 8. Update metadata
print("\n📝 Step 8: Updating document metadata...")
doc_counts = {}
for chunk in text_chunks:
    doc_id = chunk.metadata.get("doc_id", "unknown")
    doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

for doc_id, count in doc_counts.items():
    update_document_status(doc_id, "indexed", count)
    print(f"   ✅ {doc_id}: {count} chunks")

# 9. Verify
print("\n✅ Step 9: Verification...")
stats = index.describe_index_stats()
print(f"   Total vectors in index: {stats.get('total_vector_count', 0)}")
print(f"   Expected: {len(text_chunks)}")

print("\n" + "="*50)
print("🎉 CLEAN SLATE COMPLETE!")
print("="*50)