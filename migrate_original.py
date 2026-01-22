"""
One-time migration script to add doc_id to existing Medical_book.pdf
Run this ONCE before using the new system
"""

from src.document_manager import add_document, save_metadata, load_metadata

# Add original Medical_book.pdf to metadata
metadata = load_metadata()

if "original" not in metadata:
    metadata["original"] = {
        "doc_id": "original",
        "filename": "Medical_book.pdf",
        "upload_date": "2025-01-15T00:00:00",
        "status": "indexed",
        "num_chunks": 5860,
        "is_protected": True  # Cannot be deleted
    }
    save_metadata(metadata)
    print("✅ Added Medical_book.pdf as 'original' document")
    print("✅ Run /reindex endpoint to update Pinecone with doc_id metadata")
else:
    print("ℹ️  Original document already in metadata")