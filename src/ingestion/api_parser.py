import os
from google import genai
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

# List of Gemini models to try, in order of preference.
# If a model gets retired, the system automatically tries the next one.
FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

class VisionAPIParser:
    """
    [BLOCK 1: INGESTION - Robust Vision API Parser with Model Fallback]
    
    This class uses Google's Gemini Vision AI to read PDFs (even scanned images).
    Unlike traditional parsers that just look for hidden text strings, this model 
    actually "looks" at the page images and accurately reconstructs tables and 
    text into clean Markdown format that LLMs can easily understand.
    
    If a model is retired or overloaded, it automatically tries the next available model.
    """
    
    def __init__(self):
        # Load the API key from the .env file
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "CRITICAL ERROR: GEMINI_API_KEY is missing in your .env file!\n"
                "Please get one at aistudio.google.com/app/apikey"
            )
        
        # Initialize the Google GenAI client with our API key
        self.client = genai.Client(api_key=api_key)
        
    def _try_generate(self, uploaded_file, prompt: str) -> str:
        """
        Tries each model in FALLBACK_MODELS until one works.
        Returns the generated Markdown text.
        """
        last_error = None
        
        for model_name in FALLBACK_MODELS:
            try:
                print(f"  Trying model: {model_name}...")
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=[uploaded_file, prompt]
                )
                print(f"  Success with model: {model_name}")
                return response.text
            except Exception as e:
                error_msg = str(e)
                # Retry on: model retired (404), overloaded (503), rate limited (429)
                if any(code in error_msg for code in ["404", "503", "429"]):
                    print(f"  Model '{model_name}' unavailable ({error_msg[:60]}...), trying next...")
                    last_error = e
                    continue
                else:
                    # If it's a different error (auth, network etc.), don't retry
                    raise e
        
        # If we exhausted all models, raise the last error
        raise Exception(f"All fallback models failed. Last error: {last_error}")
        
    def extract_text_from_pdf(self, file_path: str) -> list[dict]:
        """
        Uploads the PDF to the Vision API, transcribes it to Markdown, 
        and returns it in our standard list-of-dictionaries format.
        """
        filename = os.path.basename(file_path)
        print(f"Uploading '{filename}' to Vision API for OCR & Layout Parsing...")
        
        try:
            # 1. Upload the file to the API's secure temporary processing storage
            uploaded_file = self.client.files.upload(file=file_path)
            
            # 2. Give the AI strict instructions on how to parse the document
            prompt = """
            You are an expert document parser. 
            Read this entire PDF document (even if it is a scanned image) and 
            convert it into clean, readable Markdown format.
            If there are tables, you MUST format them as proper Markdown tables 
            (e.g. | Column 1 | Column 2 |).
            Do not summarize the document. Transcribe the text exactly as it appears.
            """
            
            print("Processing document structure and reading tables...")
            
            # 3. Try generating with fallback models
            markdown_text = self._try_generate(uploaded_file, prompt)
            
            # 4. Delete the file from the cloud server immediately for privacy
            self.client.files.delete(name=uploaded_file.name)
            
            print("Successfully transcribed document into clean Markdown!")
            
            # 5. Return in standard format for our pipeline
            return [{
                "page_content": markdown_text,
                "metadata": {
                    "source": filename,
                    "page_number": 1 
                }
            }]
            
        except Exception as e:
            print(f"Error during API parsing: {e}")
            return []
