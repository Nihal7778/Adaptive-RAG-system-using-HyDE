from langchain_classic.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


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
    return text_chunks



    