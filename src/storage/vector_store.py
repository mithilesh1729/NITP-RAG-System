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
