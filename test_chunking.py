from src.ingestion.pdf_parser import extract_text_from_pdf
from src.processing.chunker import chunk_text

def test_text_chunking():
    pdf_path = "data/sample_notice.pdf"
    
    print("1. Extracting text from PDF (Block 1)...")
    pages = extract_text_from_pdf(pdf_path)
    
    if not pages:
        print("No text found. Please check Block 1.")
        return
        
    print("2. Chunking text (Block 2)...")
    # We use a small chunk_size (150 chars) and overlap (30 chars) 
    # so we can easily see the overlap in the terminal output!
    chunks = chunk_text(pages, chunk_size=150, chunk_overlap=30)
    
    print(f"\nSuccessfully created {len(chunks)} chunks from {len(pages)} page(s).\n")
    
    if len(chunks) >= 2:
        print("=" * 60)
        print(f"CHUNK 0 (Length: {len(chunks[0]['page_content'])} chars)")
        print(f"Metadata: {chunks[0]['metadata']}")
        print("=" * 60)
        print(chunks[0]['page_content'])
        print("\n")
        
        print("=" * 60)
        print(f"CHUNK 1 (Length: {len(chunks[1]['page_content'])} chars)")
        print(f"Metadata: {chunks[1]['metadata']}")
        print("=" * 60)
        print(chunks[1]['page_content'])
        print("=" * 60)
        
        print("\nLook closely! You should see the end of Chunk 0 repeated at the start of Chunk 1.")
    else:
        print("The text was so short it only made 1 chunk!")
        print("Here it is:")
        print(chunks[0]['page_content'])

if __name__ == "__main__":
    test_text_chunking()

