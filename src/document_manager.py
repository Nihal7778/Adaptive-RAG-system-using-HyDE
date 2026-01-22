import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional


METADATA_FILE = "Data/metadata.json"


def load_metadata() -> Dict:
    """Load metadata.json"""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_metadata(metadata: Dict):
    """Save metadata.json"""
    os.makedirs("Data", exist_ok=True)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def add_document(filename: str, is_protected: bool = False) -> str:
    """Add new document to metadata, return doc_id"""
    metadata = load_metadata()
    doc_id = str(uuid.uuid4())[:8]  # Short UUID
    
    metadata[doc_id] = {
        "doc_id": doc_id,
        "filename": filename,
        "upload_date": datetime.now().isoformat(),
        "status": "pending",
        "num_chunks": 0,
        "is_protected": is_protected
    }
    
    save_metadata(metadata)
    return doc_id


def update_document_status(doc_id: str, status: str, num_chunks: int = 0):
    """Update document status after indexing"""
    metadata = load_metadata()
    if doc_id in metadata:
        metadata[doc_id]["status"] = status
        metadata[doc_id]["num_chunks"] = num_chunks
        save_metadata(metadata)


def delete_document(doc_id: str) -> bool:
    """Remove document from metadata"""
    metadata = load_metadata()
    if doc_id in metadata:
        if metadata[doc_id].get("is_protected"):
            return False  # Cannot delete protected docs
        del metadata[doc_id]
        save_metadata(metadata)
        return True
    return False


def get_all_documents() -> List[Dict]:
    """Get list of all documents"""
    metadata = load_metadata()
    return list(metadata.values())


def get_document(doc_id: str) -> Optional[Dict]:
    """Get single document info"""
    metadata = load_metadata()
    return metadata.get(doc_id)