from flask import Flask, render_template, request, jsonify
from src.helper import download_hugging_face_embeddings, build_sources, load_pdf_data, text_split
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
from werkzeug.utils import secure_filename
from src.document_manager import add_document, delete_document, get_all_documents, update_document_status, get_document
import os
from src.hyde import HyDERetriever
import time

app = Flask(__name__)
load_dotenv()

# Upload config 
UPLOAD_FOLDER = 'Data'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Environment setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize components
embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(embedding=embeddings, index_name=index_name)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_completion_tokens=800)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
hyde_retriever = HyDERetriever()



@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return jsonify({"answer": "", "sources": []})

    doc_ids = request.form.get("doc_ids", "").strip()
    search_kwargs = {"k": 5}
    if doc_ids:
        doc_id_list = [d.strip() for d in doc_ids.split(",") if d.strip()]
        if doc_id_list:
            search_kwargs["filter"] = {"doc_id": {"$in": doc_id_list}}

    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": msg})

    answer_obj = response.get("answer") or response.get("result") or response.get("output") or response.get("text") or ""
    answer = getattr(answer_obj, "content", str(answer_obj)) if answer_obj else ""
    context_docs = response.get("context", [])
    sources = build_sources(context_docs)

    return jsonify({"answer": answer, "sources": sources})

@app.route("/documents", methods=["GET"])
def list_documents():
    return jsonify({"documents": get_all_documents()})

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400
    
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    doc_id = add_document(filename, is_protected=False)
    return jsonify({"success": True, "doc_id": doc_id, "filename": filename, "message": "Upload successful! Click 'Reindex All' to process."})

@app.route("/reindex", methods=["POST"])
def reindex_all():
    """Re-index all documents - IMPROVED VERSION"""
    try:
        from pinecone import Pinecone
        import time
        
        print("\n" + "="*50)
        print("🔄 STARTING REINDEX")
        print("="*50)
        
        # Step 1: Load PDFs
        print("\n📄 Step 1: Loading PDFs from Data/...")
        extracted_data = load_pdf_data("Data/")
        print(f"   ✅ Loaded {len(extracted_data)} document pages")
        
        # Step 2: Split into chunks
        print("\n✂️ Step 2: Splitting into chunks...")
        text_chunks = text_split(extracted_data)
        print(f"   ✅ Created {len(text_chunks)} chunks")
        
        # Step 3: Verify doc_ids
        print("\n🔍 Step 3: Verifying doc_ids...")
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
            print(f"   ⚠️ WARNING: {no_doc_id_count} chunks missing doc_id!")
            return jsonify({"error": f"{no_doc_id_count} chunks missing doc_id. Check metadata.json"}), 500
        
        # Step 4: Get embeddings
        print("\n🧠 Step 4: Loading embeddings...")
        embeddings = download_hugging_face_embeddings()
        
        # Step 5: Connect to Pinecone
        print("\n🔌 Step 5: Connecting to Pinecone...")
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(index_name)
        
        # Step 6: DELETE ALL - PROPERLY
        print("\n🗑️ Step 6: Deleting ALL existing vectors...")
        try:
            # Get current stats
            stats = index.describe_index_stats()
            total_before = stats.get('total_vector_count', 0)
            print(f"   Current vectors in index: {total_before}")
            
            if total_before > 0:
                # Method 1: Try delete with filter
                try:
                    print("   Attempting delete with empty filter...")
                    index.delete(filter={})
                    time.sleep(2)  # Wait for propagation
                except Exception as e:
                    print(f"   Filter delete failed: {e}")
                
                # Method 2: Fetch and delete by IDs
                print("   Fetching vector IDs...")
                all_ids = []
                
                # Query multiple times to get all IDs
                for attempt in range(3):  # Try 3 times to get all IDs
                    results = index.query(
                        vector=[0.0] * 384,
                        top_k=10000,
                        include_values=False,
                        include_metadata=False
                    )
                    if results and 'matches' in results:
                        batch_ids = [m['id'] for m in results['matches']]
                        all_ids.extend(batch_ids)
                        if len(batch_ids) < 10000:
                            break  # Got all
                
                # Remove duplicates
                all_ids = list(set(all_ids))
                
                if all_ids:
                    print(f"   Deleting {len(all_ids)} vectors in batches...")
                    batch_size = 1000
                    for i in range(0, len(all_ids), batch_size):
                        batch = all_ids[i:i+batch_size]
                        index.delete(ids=batch)
                        print(f"      Deleted batch {i//batch_size + 1}/{(len(all_ids)-1)//batch_size + 1}")
                    
                    # IMPORTANT: Wait for deletes to propagate
                    print("   ⏳ Waiting 5 seconds for deletions to propagate...")
                    time.sleep(5)
                    
                    # Verify deletion
                    stats_after = index.describe_index_stats()
                    remaining = stats_after.get('total_vector_count', 0)
                    print(f"   ✅ Vectors remaining: {remaining}")
                    
                    if remaining > 100:  # Some may remain due to eventual consistency
                        print(f"   ⚠️ Warning: {remaining} vectors still present (eventual consistency)")
                else:
                    print("   ℹ️ No vectors found to delete")
            else:
                print("   ℹ️ Index already empty")
                
        except Exception as e:
            print(f"   ❌ Delete failed: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Failed to clear index: {str(e)}"}), 500
        
        # Step 7: Upload new vectors
        print("\n⬆️ Step 7: Uploading new vectors...")
        print(f"   Uploading {len(text_chunks)} chunks (this takes 2-3 minutes)...")
        
        try:
            PineconeVectorStore.from_documents(
                documents=text_chunks,
                embedding=embeddings,
                index_name=index_name
            )
            print("   ✅ Upload complete!")
            
            # Wait for upload to settle
            time.sleep(2)
            
        except Exception as e:
            print(f"   ❌ Upload failed: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Failed to upload vectors: {str(e)}"}), 500
        
        # Step 8: Update metadata
        print("\n📝 Step 8: Updating document statuses...")
        doc_counts = {}
        for chunk in text_chunks:
            doc_id = chunk.metadata.get("doc_id", "unknown")
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        
        for doc_id, count in doc_counts.items():
            update_document_status(doc_id, "indexed", count)
            print(f"   ✅ {doc_id}: {count} chunks → indexed")
        
        # Step 9: Final verification
        print("\n✅ Step 9: Final verification...")
        final_stats = index.describe_index_stats()
        final_count = final_stats.get('total_vector_count', 0)
        print(f"   Total vectors in index: {final_count}")
        print(f"   Expected: {len(text_chunks)}")
        
        print("\n" + "="*50)
        print("🎉 REINDEX COMPLETE!")
        print("="*50 + "\n")
        
        return jsonify({
            "success": True,
            "message": f"Successfully reindexed {len(text_chunks)} chunks from {len(doc_counts)} documents",
            "document_counts": doc_counts,
            "vectors_in_index": final_count
        })
        
    except Exception as e:
        print(f"\n❌ REINDEX FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/delete/<doc_id>", methods=["DELETE"])
def delete_doc(doc_id):
    """Delete document and its vectors"""
    try:
        from pinecone import Pinecone
        
        # Get document info
        doc_info = get_document(doc_id)
        if not doc_info:
            return jsonify({"error": "Document not found"}), 404
        
        if doc_info.get("is_protected"):
            return jsonify({"error": "Cannot delete protected document"}), 403
        
        # Delete from Pinecone
        print(f"🗑️ Deleting vectors for doc_id: {doc_id}")
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(index_name)
        
        # Delete vectors with this doc_id using metadata filter
        try:
            index.delete(filter={"doc_id": doc_id})
            print(f"   ✅ Deleted vectors for {doc_id}")
        except Exception as e:
            print(f"   ⚠️ Pinecone delete error: {e}")
            # Continue anyway to delete file and metadata
        
        # Delete PDF file
        filepath = os.path.join("Data", doc_info["filename"])
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"   ✅ Deleted file: {filepath}")
        
        # Remove from metadata.json
        delete_document(doc_id)
        print(f"   ✅ Removed from metadata")
        
        return jsonify({
            "success": True,
            "message": f"Deleted {doc_info['filename']}"
        })
        
    except Exception as e:
        print(f"❌ Delete failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    

@app.route("/get_hyde", methods=["POST"])
def chat_hyde():
    """Chat endpoint using HyDE for improved retrieval"""
    msg = request.form.get("msg", "").strip()
    if not msg:
        return jsonify({"answer": "", "sources": [], "hypothesis": "", "method": "hyde"})

    # Get document filter
    doc_ids = request.form.get("doc_ids", "").strip()
    filter_dict = None
    if doc_ids:
        doc_id_list = [d.strip() for d in doc_ids.split(",") if d.strip()]
        if doc_id_list:
            filter_dict = {"doc_id": {"$in": doc_id_list}}
    
    # HyDE retrieval
    start_time = time.time()
    retrieved_docs, hypothesis, gen_time = hyde_retriever.retrieve_with_hyde(
        docsearch, 
        msg, 
        k=5,
        filter_dict=filter_dict
    )
    
    # Build context from retrieved docs
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Generate final answer using RAG chain
    final_prompt = f"""{system_prompt}

Context:
{context_text}

Question: {msg}

Answer:"""
    
    answer_response = llm.invoke(final_prompt)
    answer = answer_response.content
    
    sources = build_sources(retrieved_docs)
    total_time = time.time() - start_time
    
    return jsonify({
        "answer": answer,
        "sources": sources,
        "hypothesis": hypothesis,
        "method": "hyde",
        "timing": {
            "hypothesis_generation": round(gen_time, 2),
            "total_time": round(total_time, 2)
        }
    })
    


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)