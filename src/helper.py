from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
import json


#Extracting data from PDF files

def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_pdf_data(file_path):
    loader = DirectoryLoader(file_path,
                             glob ="*.pdf",
                             loader_cls= PyPDFLoader)
    

    documents = loader.load()


    return documents



def load_metadata():
    """Load metadata.json"""
    metadata_path = "Data/metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}



#Split the Data into Text Chunks

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    
    # NEW: Load metadata once outside loop
    metadata = load_metadata()

    for i, d in enumerate(text_chunks):
        meta = d.metadata or {}

        # doc name from source path (by default pypdfloader usually sets "source")
        src = meta.get("source", "")
        filename = os.path.basename(src) if src else meta.get("doc_name", "document")
        meta["doc_name"] = filename

        # NEW: Find doc_id from metadata.json (ADD THIS SECTION HERE)
        doc_id = None
        for did, info in metadata.items():
            if info.get("filename") == filename:
                doc_id = did
                break
        
        # NEW: Add doc_id to metadata
        if doc_id:
            meta["doc_id"] = doc_id
        else:
            # Fallback: if not in metadata, use filename as doc_id
            meta["doc_id"] = filename.replace(".pdf", "")

        # page is typically 0 based in loaders we convert to 1 based display
        if "page" in meta and isinstance(meta["page"], int):
            meta["page_display"] = meta["page"] + 1
        else:
            meta["page_display"] = None

        # stable chunk id (simplified version)
        meta["chunk_id"] = meta.get("chunk_id") or f"{meta.get('doc_name','doc')}_p{meta.get('page_display','?')}_c{i}"

        d.metadata = meta

    return text_chunks


def build_sources(context_docs):
    """
    Return clean, deduped sources as:
    'filename.pdf — Page N'
    """
    sources = []
    seen = set()

    for d in context_docs or []:
        meta = getattr(d, "metadata", {}) or {}

        # doc name
        src = meta.get("source", "")
        doc_name = meta.get("doc_name") or (os.path.basename(src) if src else "document.pdf")

        # page number (prefer page_display if you added it)
        page = meta.get("page_display")
        if page is None and isinstance(meta.get("page"), int):
            page = meta["page"] + 1  # convert 0-based to 1-based

        # Build a clean label
        if page:
            key = (doc_name, page)
            label = f"{doc_name} . Page {page}"
        else:
            key = (doc_name, None)
            label = f"{doc_name}"

        # Deduplicate
        if key not in seen:
            seen.add(key)
            sources.append(label)

    return sources

    