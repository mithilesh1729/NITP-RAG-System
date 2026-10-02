import os
import pymupdf4llm

class LocalPDFParser:
    """
    [100% OFFLINE LOCAL PARSER]
    Because Google's Cloud API blocked us with a 24-hour quota limit,
    we are switching to a powerful local library (pymupdf4llm) that converts 
    PDFs directly to Markdown on your own CPU! 
    
    Zero internet required. Zero API keys. Zero quotas.
    """
    
    def extract_text_from_pdf(self, file_path: str) -> list[dict]:
        filename = os.path.basename(file_path)
        print(f"Locally extracting Markdown from '{filename}' (No Cloud API!)...")
        
        try:
            # pymupdf4llm is a state-of-the-art local library for LLM parsing
            markdown_text = pymupdf4llm.to_markdown(file_path)
            
            print("Successfully extracted Markdown locally!")
            
            return [{
                "page_content": markdown_text,
                "metadata": {
                    "source": filename,
                    "page_number": 1 
                }
            }]
            
        except Exception as e:
            print(f"Error during local parsing: {e}")
            return []
