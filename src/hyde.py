"""
HyDE (Hypothetical Document Embeddings) for Medical RAG
Improves retrieval by generating hypothetical medical answers first.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any, Tuple
import time


class HyDERetriever:
    """
    Retrieves documents using hypothetical document embeddings.
    Converts user queries into medical terminology before retrieval.
    """
    
    def __init__(self, llm_model: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initialize HyDE retriever.
        
        Args:
            llm_model: OpenAI model for hypothesis generation
            temperature: Higher for diverse hypotheses (0.5-0.8 recommended)
        """
        self.llm = ChatOpenAI(
            model=llm_model, 
            temperature=temperature, 
            max_tokens=400
        )
        
        # Prompt template for hypothesis generation
        self.prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are a medical expert writing a textbook passage. 
Generate a factual medical answer to this question using proper medical terminology.

Question: {question}

Write a concise medical passage (2-3 sentences) that would appear in a medical reference:"""
        )
        
        # Create chain
        self.hyde_chain = self.prompt | self.llm | StrOutputParser()
    
    def generate_hypothesis(self, question: str) -> str:
        """
        Generate hypothetical medical answer.
        
        Args:
            question: User's question
            
        Returns:
            Hypothetical answer with medical terminology
        """
        try:
            hypothesis = self.hyde_chain.invoke({"question": question})
            return hypothesis.strip()
        except Exception as e:
            print(f"⚠️ HyDE generation failed: {e}")
            return question  # Fallback to original question
    
    def retrieve_with_hyde(
    self, 
    docsearch, 
    question: str, 
    k: int = 5,
    filter_dict: Dict[str, Any] =  None
) -> Tuple[List, str, float]:
        """
        Retrieve documents using HyDE with multi-query merging.
        Searches with BOTH original question and hypothesis, then merges results.
        
        Args:
        docsearch: Pinecone vector store
        question: User question
        k: Number of documents to retrieve
        filter_dict: Optional metadata filter for doc_id
        
        Returns:
            (retrieved_docs, hypothesis, generation_time)
        """
        # Generate hypothesis
        start_time = time.time()
        hypothesis = self.generate_hypothesis(question)
        gen_time = time.time() - start_time
        
        # DUAL RETRIEVAL: Search with both hypothesis AND original question
        search_kwargs = {"k": k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        
        # Retrieve using hypothesis
        hyde_docs = docsearch.similarity_search(hypothesis, **search_kwargs)
        
        # Retrieve using original question
        baseline_docs = docsearch.similarity_search(question, **search_kwargs)
        
        # Merge and deduplicate by chunk_id
        seen_chunks = set()
        merged_docs = []
        
        for doc in hyde_docs + baseline_docs:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                merged_docs.append(doc)
            elif not chunk_id:  # If no chunk_id, add anyway
                merged_docs.append(doc)
        
        # Return top-k from merged results
        final_docs = merged_docs[:k]
        
        return final_docs, hypothesis, gen_time

