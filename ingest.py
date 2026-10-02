import os
import glob
from src.ingestion.local_parser import LocalPDFParser
from src.processing.chunker import chunk_text
from src.embeddings.generator import LocalEmbedder
from src.storage.vector_store import ChromaVectorStore

def ingest_all_documents(data_dir: str = "data"):
    print("=== STARTING 100% LOCAL INGESTION PIPELINE ===")
    
    parser = LocalPDFParser()
    embedder = LocalEmbedder()
    db = ChromaVectorStore()
    
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
        chunks = chunk_text(pages, chunk_size=4000, chunk_overlap=400)
        
        # 3. Embed
        texts_to_embed = [chunk["page_content"] for chunk in chunks]
        print("Generating math vectors for chunks...")
        embeddings = embedder.embed_batch(texts_to_embed)
        
        # 4. Store
        db.add_chunks(chunks=chunks, embeddings=embeddings)
        print(f"Successfully added {os.path.basename(pdf_path)} to database.")
        
    print("\n=== INGESTION COMPLETE ===")
    print("All documents are now in ChromaDB. You can now run 'python chat.py' to ask questions!")

if __name__ == "__main__":
    ingest_all_documents()

