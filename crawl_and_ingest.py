"""
Crawl & Ingest Pipeline
=======================
Given ANY URL containing PDFs, this script:
1. Auto-detects the best crawling strategy (API / HTML / Google Drive)
2. Downloads new PDFs to data/crawled/
3. Parses, chunks, embeds, and upserts them into ChromaDB + BM25

Usage:
    python crawl_and_ingest.py https://www.nitp.ac.in/Notices/Academic 2
    python crawl_and_ingest.py https://www.nitp.ac.in/Others/NITAct
    python crawl_and_ingest.py https://www.nitp.ac.in/Department/CSE/syllabus/btech-ai-ds
    python crawl_and_ingest.py https://any-website.com/documents
"""

import sys
import os
import re

from src.crawler.spider import UniversalSpider
from src.ingestion.local_parser import LocalPDFParser
from src.ingestion.api_parser import VisionAPIParser
from src.processing.chunker import chunk_text
from src.embeddings.generator import GeminiEmbedder
from src.storage.vector_store import ChromaVectorStore


def crawl_and_ingest(url: str, days: int = 2, max_files: int = 50):
    """Full pipeline: crawl -> parse -> chunk -> embed -> upsert."""
    
    # --- Step 1: Crawl & Download ---
    spider = UniversalSpider()
    new_files = spider.crawl(url, days=days, max_files=max_files)
    
    if not new_files:
        print("\nNothing new to ingest. System is already up to date!")
        return {
            "status": "up_to_date",
            "new_files_count": 0,
            "processed_count": 0,
            "new_chunks_count": 0,
            "files": []
        }
    
    # --- Step 2: Initialize pipeline components ---
    local_parser = LocalPDFParser()
    embedder = GeminiEmbedder()
    db = ChromaVectorStore()
    
    total_chunks = 0
    successful = 0
    
    for file_info in new_files:
        filepath = file_info["filepath"]
        filename = os.path.basename(filepath)
        print(f"\n{'─'*50}")
        print(f"Processing: {filename}")
        print(f"Title: {file_info.get('title', 'N/A')[:80]}")
        print(f"{'─'*50}")
        
        # --- Step 3: Parse PDF ---
        pages = local_parser.extract_text_from_pdf(filepath)
        
        # Check if scanned (< 50 chars) OR font corrupted (0 dictionary words extracted)
        raw_combined = " ".join(p.get("page_content", "") for p in pages)
        common_vocab = {"the", "and", "examination", "viva", "voce", "student", "roll", "department", "scholar", "supervision", "held", "notice", "phd", "patna", "date", "dr", "prof"}
        tokens = set(re.findall(r'[a-zA-Z]{3,}', raw_combined.lower()))
        vocab_hits = len(tokens.intersection(common_vocab))
        
        if len(raw_combined.strip()) < 50 or vocab_hits < 2:
            print("  [!] Scrambled font / scanned image detected (0 dictionary words). Using Gemini Vision OCR...")
            try:
                vision_parser = VisionAPIParser()
                pages = vision_parser.extract_text_from_pdf(filepath)
            except Exception as e:
                print(f"  [ERROR] Vision OCR failed: {e}")
                continue
        
        if not pages:
            print("  [SKIP] No text could be extracted.")
            continue
        
        # Inject web metadata
        for page in pages:
            page["metadata"]["category"] = file_info.get("category", "Notice")
            page["metadata"]["web_title"] = file_info.get("title", "")
            page["metadata"]["source_url"] = file_info.get("source_url", "")
            if "date" in file_info:
                page["metadata"]["notice_date"] = file_info["date"]
            if "notice_id" in file_info:
                page["metadata"]["notice_id"] = file_info["notice_id"]
            if "venue" in file_info:
                page["metadata"]["venue"] = file_info["venue"]
            if "department" in file_info:
                page["metadata"]["department"] = file_info["department"]
        
        # --- Step 4: Chunk ---
        chunks = chunk_text(pages, chunk_size=4000, chunk_overlap=400)
        print(f"  Chunks created: {len(chunks)}")
        
        if not chunks:
            continue
        
        # --- Step 5: Embed ---
        texts_to_embed = [c["page_content"] for c in chunks]
        print(f"  Embedding {len(texts_to_embed)} chunks with gemini-embedding-2...")
        embeddings = embedder.embed_batch(texts_to_embed)
        
        # --- Step 6: Upsert into ChromaDB ---
        db.add_chunks(chunks=chunks, embeddings=embeddings)
        total_chunks += len(chunks)
        successful += 1
        print(f"  Upserted {len(chunks)} chunks into ChromaDB.")
    
    # --- Step 7: Force BM25 rebuild on next query ---
    if hasattr(db, 'bm25'):
        delattr(db, 'bm25')
        print("\n[System] BM25 index will rebuild on next query to include new documents.")
    
    print(f"\n{'='*60}")
    print(f"  CRAWL & INGEST COMPLETE")
    print(f"  New PDFs processed: {successful}/{len(new_files)}")
    print(f"  New chunks indexed: {total_chunks}")
    print(f"{'='*60}")

    return {
        "status": "success",
        "new_files_count": len(new_files),
        "processed_count": successful,
        "new_chunks_count": total_chunks,
        "files": [os.path.basename(f["filepath"]) for f in new_files]
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python crawl_and_ingest.py <URL> [days] [max_files]")
        print("\nExamples:")
        print("  python crawl_and_ingest.py https://www.nitp.ac.in/Notices/Academic 2")
        print("  python crawl_and_ingest.py https://www.nitp.ac.in/Others/NITAct")
        print("  python crawl_and_ingest.py https://www.nitp.ac.in/Department/CSE/syllabus/btech-ai-ds")
        sys.exit(1)
    
    target_url = sys.argv[1]
    target_days = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    target_max = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    print(f"Target:    {target_url}")
    print(f"Lookback:  {target_days} days")
    print(f"Max files: {target_max}")
    
    crawl_and_ingest(target_url, days=target_days, max_files=target_max)
