"""
Reset Crawled Data & Revert to Baseline
========================================
Removes all downloaded files from data/crawled/, resets crawler_manifest.json,
and deletes all crawled chunks from ChromaDB, reverting the database
back to the original pristine baseline (76 chunks from 10 circulars).
"""

import os
import glob
import json
from src.storage.vector_store import ChromaVectorStore

BASELINE_SOURCES = {
    'FEE_MCA_REGULAR.pdf',
    'FEE_MCA_SELF_FINANCE.pdf',
    'FEE_PG_2024_25.pdf',
    'Fee_structure.pdf',
    'Holiday_Academic.pdf',
    'JRF_Advertisement_2026.pdf',
    'MSE July Dec 2026 Schedule _Reg.xlsx.pdf',
    'NIRF_Data_2026.pdf',
    'PhD_fee_Structure_29_07_2026.pdf',
    'UG Regulation 2022.pdf'
}

def reset():
    print("=" * 60)
    print("  RESETTING CRAWLED DATA & REVERTING TO BASELINE")
    print("=" * 60)
    
    # 1. Connect to ChromaDB
    db = ChromaVectorStore()
    all_data = db.collection.get()
    total_before = len(all_data['ids'])
    
    crawled_ids = []
    for doc_id, meta in zip(all_data['ids'], all_data['metadatas']):
        source = meta.get('source', '')
        if source not in BASELINE_SOURCES:
            crawled_ids.append(doc_id)
            
    print(f"\n1. ChromaDB Status:")
    print(f"   - Total chunks before: {total_before}")
    print(f"   - Crawled chunks to remove: {len(crawled_ids)}")
    
    if crawled_ids:
        db.collection.delete(ids=crawled_ids)
        print(f"   -> Successfully deleted {len(crawled_ids)} crawled chunks.")
        
    remaining = db.collection.count()
    print(f"   - Baseline chunks remaining: {remaining} (Expected: 76)")
    
    # 2. Reset in-memory BM25
    if hasattr(db, 'bm25'):
        delattr(db, 'bm25')
        print("   -> Reset in-memory BM25 index.")
        
    # 3. Clean up data/crawled/ directory
    crawled_dir = "data/crawled"
    deleted_files = 0
    if os.path.exists(crawled_dir):
        for f in glob.glob(os.path.join(crawled_dir, "*")):
            try:
                os.remove(f)
                deleted_files += 1
            except Exception as e:
                print(f"   [WARN] Could not remove {f}: {e}")
    print(f"\n2. Filesystem Cleaned:")
    print(f"   - Deleted {deleted_files} files from {crawled_dir}/")
    
    # 4. Reset crawler_manifest.json
    manifest_path = "data/crawler_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"downloaded": {}}, f, indent=2)
    print(f"\n3. Manifest Reset:")
    print(f"   - Reset {manifest_path} to empty registry.")
    
    print("\n" + "=" * 60)
    print("  RESET COMPLETE — System is clean and ready for fresh testing!")
    print("=" * 60)

if __name__ == "__main__":
    reset()
