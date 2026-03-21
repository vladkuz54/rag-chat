from typing import List, Dict
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
import numpy as np


class HybridSearcher:
    """Combines BM25 keyword search with vector semantic search using RRF."""
    
    def __init__(self):
        self.bm25_index = None
        self.documents = []
        self.vectorizer = None
    
    def build_bm25_index(self, documents: List[Document]):
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of Document objects to index
        """
        print(f"Building BM25 index for {len(documents)} documents...")
        self.documents = documents

        tokenized_docs = [
            doc.page_content.lower().split() 
            for doc in documents
        ]
        
        # Create BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs)
        print(f"BM25 index built successfully")
    
    def bm25_search(self, query: str, k: int = 5) -> List[tuple]:
        """
        Search using BM25 (keyword-based).
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if self.bm25_index is None:
            return []
        
 
        tokenized_query = query.lower().split()
 
        scores = self.bm25_index.get_scores(tokenized_query)
       
        top_k_idx = np.argsort(scores)[-k:][::-1]
        results = [
            (self.documents[idx], scores[idx])
            for idx in top_k_idx 
            if scores[idx] > 0
        ]
        
        return results
    
    @staticmethod
    def reciprocal_rank_fusion(
        bm25_results: List[tuple],
        vector_results: List[tuple],
        k: int = 60
    ) -> List[Document]:
        """
        Combine BM25 and vector search results using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score = sum(1 / (k + rank)) for each ranking
        
        Args:
            bm25_results: List of (document, bm25_score) tuples from BM25 search
            vector_results: List of (document, vector_score) tuples from vector search
            k: RRF parameter (typically 60)
            
        Returns:
            Ranked list of unique documents
        """
        rrf_scores: Dict[str, float] = {}
 
        for rank, (doc, _) in enumerate(bm25_results, start=1):
            doc_id = id(doc) 
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
        
       
        for rank, (doc, _) in enumerate(vector_results, start=1):
            doc_id = id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
        
        sorted_results = sorted(
            [(doc, score) for doc, score in [
                (doc, rrf_scores.get(id(doc), 0))
                for doc in [r[0] for r in bm25_results + vector_results]
                if id(doc) in rrf_scores
            ]],
            key=lambda x: x[1],
            reverse=True
        )
        
        seen = set()
        unique_results = []
        for doc, score in sorted_results:
            doc_hash = hash(doc.page_content)
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique_results.append(doc)
        
        return unique_results
    
    def hybrid_search(
        self,
        query: str,
        vector_retriever,
        k: int = 5
    ) -> List[Document]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query: Query string
            vector_retriever: LangChain vector retriever
            k: Number of results per method (total may be higher due to deduplication)
            
        Returns:
            List of combined and ranked documents
        """

        bm25_results = self.bm25_search(query, k=k)
        print(f"   BM25: Found {len(bm25_results)} results")

        vector_results = vector_retriever.invoke(query)
        vector_results_with_scores = [
            (doc, 1.0) for doc in vector_results  
        ]
        print(f"   Vector: Found {len(vector_results_with_scores)} results")
        
 
        combined_results = self.reciprocal_rank_fusion(
            bm25_results,
            vector_results_with_scores
        )
        

        return combined_results[:k * 2]  


_global_hybrid_searcher = None


def get_hybrid_searcher() -> HybridSearcher:
    """Get the global hybrid searcher instance."""
    global _global_hybrid_searcher
    if _global_hybrid_searcher is None:
        _global_hybrid_searcher = HybridSearcher()
    return _global_hybrid_searcher


def set_hybrid_searcher(searcher: HybridSearcher):
    """Set the global hybrid searcher instance."""
    global _global_hybrid_searcher
    _global_hybrid_searcher = searcher
