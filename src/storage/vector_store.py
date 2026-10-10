import os
import chromadb
from typing import List, Dict, Any

class ChromaVectorStore:
    """
    A wrapper around ChromaDB to store and retrieve document chunks.
    """
    
    def __init__(self, persist_directory: str = "vector_db", collection_name: str = "nitp_documents"):
        # Ensure the directory exists on disk
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize the persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create a collection (like a table in a SQL database)
        # We use 'cosine' similarity as our distance metric (standard for text embeddings)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} 
        )
        print(f"Connected to ChromaDB collection: '{collection_name}' at '{persist_directory}/'")

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Adds text chunks, their metadata, and their embeddings to the database.
        Uses upsert so re-ingesting the same PDF updates existing chunks 
        instead of crashing with duplicate ID errors.
        """
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []
        
        # Prepare the data in the specific format ChromaDB expects
        for i, chunk in enumerate(chunks):
            # Create a unique ID for each chunk using filename, page, and chunk index
            source = chunk["metadata"].get("source", "unknown")
            page_num = chunk["metadata"].get("page_number", "unknown")
            chunk_index = chunk["metadata"].get("chunk_index", i)
            chunk_id = f"{source}_page_{page_num}_chunk_{chunk_index}"
            
            ids.append(chunk_id)
            documents.append(chunk["page_content"])
            metadatas.append(chunk["metadata"])
            
        # Upsert (insert or update) — safe for re-ingestion
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"Successfully upserted {len(chunks)} chunks to the database.")
        
    def search(self, query_embedding: List[float], n_results: int = 3) -> dict:
        """
        Searches the database for the chunks most similar to the query embedding.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

    def hybrid_search(self, query_text: str, query_embedding: List[float], n_results: int = 3) -> dict:
        """
        M.TECH THESIS ALGORITHM: Hybrid Search with Reciprocal Rank Fusion (RRF).
        1. Dense Vector Search (ChromaDB) -> Finds "Meaning"
        2. Sparse Search (BM25) -> Finds "Exact Keywords"
        3. RRF -> Merges the two rankings to get the absolute best results.
        """
        from rank_bm25 import BM25Okapi
        import numpy as np
        
        # --- 1. Lazy load all documents for BM25 (runs once in memory) ---
        if not hasattr(self, 'bm25'):
            print("\n[System] Initializing BM25 Keyword Engine...")
            import re
            all_data = self.collection.get()
            self.all_ids = all_data['ids']
            self.all_docs = all_data['documents']
            self.all_metadatas = all_data['metadatas']
            
            # Tokenize documents properly: remove HTML tags, punctuation, and STOPWORDS
            def tokenize(text: str):
                import re
                clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
                tokens = clean_text.split()
                stopwords = {"who", "is", "the", "of", "a", "an", "what", "where", "how", "in", "on", "at", "to", "for"}
                return [t for t in tokens if t not in stopwords]
                
            self.tokenize_fn = tokenize
            tokenized_corpus = [self.tokenize_fn(doc) for doc in self.all_docs]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
        # --- 2. Get BM25 (Keyword) Rankings ---
        tokenized_query = self.tokenize_fn(query_text)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top 30 keyword matches
        k_retrieve = min(30, len(self.all_ids))
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:k_retrieve]
        bm25_ids = [self.all_ids[i] for i in bm25_top_indices]
        
        # --- 3. Get Vector (Meaning) Rankings from ChromaDB ---
        chroma_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k_retrieve
        )
        dense_ids = chroma_results['ids'][0]
        
        # --- 4. Reciprocal Rank Fusion (RRF) ---
        # Formula: RRF Score = 1 / (k_constant + rank)
        k_constant = 60
        rrf_scores = {}
        
        for rank, doc_id in enumerate(dense_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (k_constant + rank + 1))
            
        for rank, doc_id in enumerate(bm25_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (k_constant + rank + 1))
            
        # Sort by final RRF score
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_ids = [x[0] for x in sorted_rrf[:n_results]]
        
        # --- 5. Return the Winning Documents ---
        final_docs = []
        final_metadatas = []
        final_scores_info = []
        
        for i, doc_id in enumerate(top_ids):
            idx = self.all_ids.index(doc_id)
            final_docs.append(self.all_docs[idx])
            final_metadatas.append(self.all_metadatas[idx])
            
            # Calculate debug info
            bm25_rank = bm25_ids.index(doc_id) + 1 if doc_id in bm25_ids else "N/A"
            dense_rank = dense_ids.index(doc_id) + 1 if doc_id in dense_ids else "N/A"
            final_scores_info.append(f"Rank {i+1} | RRF Score: {rrf_scores[doc_id]:.4f} (Vector Rank: {dense_rank}, BM25 Rank: {bm25_rank}) | {doc_id}")
            
        return {
            "documents": [final_docs],
            "metadatas": [final_metadatas],
            "hybrid_scores": [final_scores_info]
        }
