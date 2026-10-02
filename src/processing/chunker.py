import re

def chunk_text(pages: list[dict], chunk_size: int = 2000, chunk_overlap: int = 200) -> list[dict]:
    """
    [BLOCK 8: SEMANTIC MARKDOWN CHUNKING]
    
    Unlike the old 'meat cleaver' approach that chopped text exactly at N characters,
    this smart chunker reads the Markdown structure. It splits text at logical boundaries
    like Headers (##) and Paragraphs (\n\n). 
    
    If a table is found, it keeps the ENTIRE table together in one chunk, 
    so Llama-3 never loses the column headers!
    """
    chunks = []
    
    # We split by major structural boundaries: 
    # - New Markdown headers (\n#)
    # - Double newlines which separate paragraphs/tables (\n\n)
    split_pattern = r"(?=\n# |\n## |\n\n)"
    
    for page in pages:
        text = page["page_content"]
        original_metadata = page["metadata"]
        
        if not text:
            continue
            
        # Split the text semantically based on the pattern
        semantic_blocks = re.split(split_pattern, text)
        
        current_chunk_text = ""
        chunk_index = 0
        
        for block in semantic_blocks:
            block = block.strip()
            if not block:
                continue
                
            # If adding this structural block exceeds our target size, save the current chunk!
            if len(current_chunk_text) + len(block) > chunk_size and current_chunk_text:
                
                chunk_metadata = original_metadata.copy()
                chunk_metadata["chunk_index"] = chunk_index
                
                chunks.append({
                    "page_content": current_chunk_text.strip(),
                    "metadata": chunk_metadata
                })
                
                # Start a fresh chunk with the new block
                current_chunk_text = block
                chunk_index += 1
            else:
                # Keep accumulating blocks into the current chunk
                current_chunk_text += "\n\n" + block if current_chunk_text else block
                
        # Save the very last chunk
        if current_chunk_text:
            chunk_metadata = original_metadata.copy()
            chunk_metadata["chunk_index"] = chunk_index
            chunks.append({
                "page_content": current_chunk_text.strip(),
                "metadata": chunk_metadata
            })
            
    return chunks