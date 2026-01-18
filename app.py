from flask import Flask, render_template, request, jsonify
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from src.helper import build_sources
import os


app = Flask(__name__)

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"


docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings, 
    index_name=index_name
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
        
    ]
)

question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return jsonify({"answer": "", "sources": []})

    response = rag_chain.invoke({"input": msg})

    answer_obj = (
        response.get("answer")
        or response.get("result")
        or response.get("output")
        or response.get("text")
        or ""
    )
    answer = answer_obj.content if hasattr(answer_obj, "content") else str(answer_obj)

    context_docs = response.get("context", [])
    sources = build_sources(context_docs)

    return jsonify({"answer": answer, "sources": sources})




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)  


    

