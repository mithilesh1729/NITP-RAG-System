import os
from src.ingestion.pdf_parser import extract_text_from_pdf
from src.processing.chunker import chunk_text
from src.embeddings.generator import LocalEmbedder
from src.storage.vector_store import ChromaVectorStore

def test_vector_storage():
    pdf_path = "data/sample_notice.pdf"
    
    print("=== PIPELINE START ===")
    
    # 1. Extraction
    print("\n1. Extracting text...")
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        print("No text found. Aborting.")
        return
        
    # 2. Chunking
    print("\n2. Chunking text...")
    # We use a standard, realistic chunk size here
    chunks = chunk_text(pages, chunk_size=500, chunk_overlap=50)
    print(f"Created {len(chunks)} chunks.")
    
    # 3. Embedding
    print("\n3. Loading Embedder & Generating Vectors...")
    embedder = LocalEmbedder()
    
    # Extract just the raw text strings for the embedder
    texts_to_embed = [chunk["page_content"] for chunk in chunks]
    
    # Generate embeddings for all chunks (batching is faster!)
    embeddings = embedder.embed_batch(texts_to_embed)
    print(f"Generated {len(embeddings)} vectors.")
    
    # 4. Storage
    print("\n4. Initializing Vector Database...")
    db = ChromaVectorStore()
    
    print("Saving chunks to database...")
    db.add_chunks(chunks=chunks, embeddings=embeddings)
    
    # 5. Quick Test Search
    print("\n=== TEST SEARCH ===")
    # test_query = "fee component for Dual Degree program"
    
    test_query="What is the Name of Registrar who signed the document?"
    print(f"User Query: '{test_query}'")
    
    # IMPORTANT: We MUST embed the user's query using the EXACT SAME model we used for the chunks!
    query_vector = embedder.embed_text(test_query)
    
    print("Searching database for nearest neighbors...")
    # Find the top 1 closest match
    results = db.search(query_embedding=query_vector, n_results=1)
    
    print("\n=== TOP MATCH FOUND ===")
    # Chroma returns a slightly nested dictionary structure.
    # We grab the first result [0] of the first query [0].
    if results['documents'] and len(results['documents'][0]) > 0:
        top_document = results['documents'][0][0]
        top_metadata = results['metadatas'][0][0]
        distance = results['distances'][0][0] # Lower distance = higher similarity
        
        print(f"Metadata: {top_metadata}")
        print(f"Cosine Distance: {round(distance, 4)} (Lower is better)")
        print("-" * 50)
        print(top_document)
        print("-" * 50)
    else:
        print("No results found.")

if __name__ == "__main__":
    test_vector_storage()

