import os
from src.ingestion.pdf_parser import extract_text_from_pdf

def test_pdf_extraction():
    # 1. Define the path to our data directory and a test PDF
    data_dir = "data"
    test_pdf_name = "sample_notice.pdf"
    test_pdf_path = os.path.join(data_dir, test_pdf_name)
    
    # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # 2. Check if the user has provided a test PDF
    if not os.path.exists(test_pdf_path):
        print(f"--- Action Required ---")
        print(f"Please place a PDF file named '{test_pdf_name}' inside the '{data_dir}/' folder.")
        print(f"Current absolute path expected: {os.path.abspath(test_pdf_path)}")
        return
        
    print(f"Found test PDF at: {test_pdf_path}")
    print("Extracting text...\n")
    
    # 3. Call our extraction function
    pages = extract_text_from_pdf(test_pdf_path)
    
    # 4. Print the results
    if not pages:
        print("No text could be extracted or the file is empty/invalid.")
        return
        
    print(f"Successfully extracted {len(pages)} pages.\n")
    
    # Print the first 500 characters of the first page to verify
    first_page = pages[0]
    content = first_page["page_content"]
    metadata = first_page["metadata"]
    
    print("-" * 50)
    print(f"Metadata for Page 1: {metadata}")
    print("-" * 50)
    print("Content preview (first 500 characters):")
    print("-" * 50)
    print(content[:500] + ("..." if len(content) > 500 else ""))
    print("-" * 50)

if __name__ == "__main__":
    test_pdf_extraction()

