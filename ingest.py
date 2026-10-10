import os
import glob
import shutil
from src.ingestion.local_parser import LocalPDFParser
from src.processing.chunker import chunk_text
from src.embeddings.generator import GeminiEmbedder
from src.storage.vector_store import ChromaVectorStore

def ingest_all_documents(data_dir: str = "data"):
    print("=== STARTING CLOUD-EMBEDDING INGESTION PIPELINE ===")
    
    db_path = "vector_db"
    # if os.path.exists(db_path):
    #     print("Clearing old vector database due to Embedding Model upgrade...")
    #     try:
    #         shutil.rmtree(db_path)
    #     except PermissionError:
    #         print("WARNING: Could not delete old database. Is chat.py running? Close it first!")
    #         return
            
    parser = LocalPDFParser()
    embedder = GeminiEmbedder()
    db = ChromaVectorStore(persist_directory=db_path)
    
    # Find all PDFs in the data folder
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in the '{data_dir}' folder.")
        return
        
    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path}")
        
        # 1. Extract
        pages = parser.extract_text_from_pdf(pdf_path)
        if not pages:
            print(f"Skipping {pdf_path} due to parsing error.")
            continue
            
        # 2. Chunk
        # Using 4000 chars so tables stay completely intact!
        # Gemini Embedder can handle up to ~8000 chars without truncation.
        chunks = chunk_text(pages, chunk_size=4000, chunk_overlap=400)
        
        # 3. Embed
        texts_to_embed = [chunk["page_content"] for chunk in chunks]
        print(f"Generating math vectors for {len(chunks)} chunks using Gemini API...")
        embeddings = embedder.embed_batch(texts_to_embed)
        
        # 4. Store
        db.add_chunks(chunks=chunks, embeddings=embeddings)
        print(f"Successfully added {os.path.basename(pdf_path)} to database.")
        
    print("\n=== INGESTION COMPLETE ===")
    print("All documents are now in ChromaDB. You can now run 'python chat.py' to ask questions!")

if __name__ == "__main__":
    ingest_all_documents()
