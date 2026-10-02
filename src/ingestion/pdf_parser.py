import os
from pypdf import PdfReader

def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Reads a PDF file and extracts text page by page.
    
    Args:
        file_path (str): The path to the PDF file.
        
    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains
                    'page_content' (the extracted text) and 
                    'metadata' (source file name and page number).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    extracted_pages = []
    
    try:
        # Open the PDF file
        reader = PdfReader(file_path)
        
        # Get the filename for metadata tracking
        filename = os.path.basename(file_path)
        
        # Iterate through all pages
        for page_num, page in enumerate(reader.pages):
            # Extract text from the page
            text = page.extract_text()
            
            # Clean up the text slightly (remove extra spaces/newlines)
            if text:
                text = text.strip()
            else:
                text = ""  # Handle empty pages safely
                
            # Create a dictionary for this page
            page_data = {
                "page_content": text,
                "metadata": {
                    "source": filename,
                    "page_number": page_num + 1  # 1-indexed for human readability
                }
            }
            extracted_pages.append(page_data)
            
        return extracted_pages
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

