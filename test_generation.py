import os
from src.ingestion.api_parser import VisionAPIParser
from src.processing.chunker import chunk_text
from src.embeddings.generator import LocalEmbedder
from src.storage.vector_store import ChromaVectorStore
from src.generation.llm import LocalLLM

def test_full_rag_pipeline():
    pdf_path = "data/viva.pdf"
    print("=== STARTING FULL LOCAL RAG PIPELINE ===")
    
    # --- PHASE 1: INDEXING ---
    print("\n[Phase 1: Indexing - Robust Vision API]")
    parser = VisionAPIParser()
    pages = parser.extract_text_from_pdf(pdf_path)
    
    # Stop immediately if parsing failed (bad API key, network error, etc.)
    if not pages:
        print("FATAL: PDF parsing returned no content. Fix the error above and retry.")
        return
    
    # CRITICAL: We use large chunks to keep Markdown tables intact.
    # If a table is sliced in half by a small chunk, the bottom rows
    # lose their column headers, making the numbers meaningless!
    chunks = chunk_text(pages, chunk_size=4000, chunk_overlap=400)
    
    embedder = LocalEmbedder()
    texts_to_embed = [chunk["page_content"] for chunk in chunks]
    embeddings = embedder.embed_batch(texts_to_embed)
    
    db = ChromaVectorStore()
    db.add_chunks(chunks=chunks, embeddings=embeddings)
    
    
    
    
    # --- PHASE 2: RETRIEVAL ---
    print("\n[Phase 2: Retrieval]")
    # Let's ask a question that is answered in plain text, not inside a messy table!
    test_query = "Who will be the faculty members for my End Semester viva for Database Management System, if I am MCA DS&I student ?"
    
    print(f"User Query: '{test_query}'")
    query_vector = embedder.embed_text(test_query)
    
    # Since our chunks are now massive (4000 chars), we only retrieve the top 2 
    # to avoid blowing up the LLM's memory limit (Context Window).
    results = db.search(query_embedding=query_vector, n_results=2)
    
    retrieved_text = ""
    sources = set()
    
    if results['documents'] and len(results['documents'][0]) > 0:
        # Join the chunks together for the LLM
        retrieved_text = "\n\n".join(results['documents'][0])
        
        # Deterministically extract the exact document names from the database metadata
        for metadata in results['metadatas'][0]:
            sources.add(metadata.get('source', 'Unknown Document'))
            
        print(f"Found {len(results['documents'][0])} relevant chunks in the database.")
        print("Raw Context retrieved length:", len(retrieved_text), "characters.")
    else:
        print("No context found. Aborting.")
        return






    # --- PHASE 3: GENERATION ---
    print("\n[Phase 3: Generation - Cloud LLM]")
    print("Sending prompt to Gemini (since local Llama-3 struggles with huge tables)...")
    
    from src.generation.llm import GeminiLLM
    llm = GeminiLLM() 
    
    final_answer = llm.generate_answer(query=test_query, context=retrieved_text)
    
    print("\n=== FINAL AI ANSWER ===")
    print(final_answer)
    
    # ENGINEERING TRADEOFF: 
    # We do not ask the LLM to write the citation. LLMs hallucinate file names.
    # Instead, we programmatically append the verified database metadata here.
    if "I don't know" not in final_answer:
        print("\n[Sources used:]")
        for source in sources:
            print(f"- {source}")
            
    print("=======================")

if __name__ == "__main__":
    test_full_rag_pipeline()

