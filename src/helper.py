from langchain_classic.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os


#Extracting data from PDF files

def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_pdf_data(file_path):
    loader = DirectoryLoader(file_path,
                             glob ="*.pdf",
                             loader_cls=PyPDFLoader)
    

    documents = loader.load()


    return documents




#Split the Data into Text Chunks

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )
    text_chunks = text_splitter.split_documents(extracted_data)

    for i, d in enumerate(text_chunks):
        meta = d.metadata or {}

        # doc name from source path (by default pypdfloader usually sets "source")
        src = meta.get("source", "")
        meta["doc_name"] = os.path.basename(src) if src else meta.get("doc_name", "document")

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
    sources = []
    seen = set()

    for d in context_docs or []:
        meta = getattr(d, "metadata", {}) or {}
        doc_name = meta.get("doc_name") or os.path.basename(str(meta.get("source", "document")))
        page = meta.get("page_display") or (meta.get("page") + 1 if isinstance(meta.get("page"), int) else None)
        chunk_id = meta.get("chunk_id")

        if page is not None:
            label = f"{doc_name} • p.{page}"
        else:
            label = f"{doc_name}"

        if chunk_id is not None:
            label = f"{label} • {chunk_id}"

        label = str(label)

        if label not in seen:
            seen.add(label)
            sources.append(label)

    return sources

    