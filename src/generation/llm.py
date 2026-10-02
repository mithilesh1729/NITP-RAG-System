import requests

class LocalLLM:
    """
    A simple wrapper to communicate with a local Ollama instance.
    No heavy frameworks used—just standard HTTP requests!
    """
    
    def __init__(self, model_name: str = "llama3", url: str = "http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.url = url
        
    def generate_answer(self, query: str, context: str) -> str:
        """
        Takes the user query and the retrieved context, builds a strict prompt, 
        and asks Ollama to generate an answer.
        """
        
        # 1. Build the Prompt (The most important part of RAG generation!)
        prompt = f"""You are a helpful assistant for NIT Patna. 
Answer the user's question using ONLY the provided context below.
If the answer is not contained in the context, say "I don't know based on the provided document".
Do not use outside knowledge.

Context:
{context}

Question: {query}

Answer:"""

        # 2. Prepare the payload for Ollama's API
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False # We want the full answer at once to keep it simple
        }
        
        # 3. Send the request
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status() # Raise an error if the request failed
            
            # Parse the JSON response
            result = response.json()
            return result.get("response", "No response generated.")
            
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Is Ollama currently running on your computer?"
        except Exception as e:
            return f"Error communicating with LLM: {e}"

class GeminiLLM:
    """
    A wrapper to use Google's Gemini models for Generation.
    Cloud models are infinitely better at reasoning over large, complex Markdown tables
    than small local models like Llama-3 8B.
    """
    def __init__(self):
        import os
        from google import genai
        from dotenv import load_dotenv
        
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found! Check your .env file.")
            
        self.client = genai.Client(api_key=api_key)
        
    def generate_answer(self, query: str, context: str) -> str:
        prompt = f"""You are a helpful assistant for NIT Patna. 
Answer the user's question using ONLY the provided context below.
If the answer is not contained in the context, say "I don't know based on the provided document".
Do not use outside knowledge. Pay close attention to Markdown tables and column headers!

Context:
{context}

Question: {query}

Answer:"""
        import time
        
        FALLBACK_MODELS = [
            "gemini-3.6-flash",      # Switched 3.6 to primary to avoid 2.5 daily limit
            "gemini-flash-latest",   # Generic pointer to latest available
            "gemini-2.5-flash" 
        ]
        
        for model_name in FALLBACK_MODELS:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text
            except Exception as e:
                error_msg = str(e)
                if any(code in error_msg for code in ["429", "503", "404"]):
                    print(f"\n[System: {model_name} rate-limited. Switching to next model...]")
                    continue
                else:
                    return f"Error communicating with Gemini: {error_msg}"
                    
        return "System Error: All AI models are currently rate-limited. Please wait 1 minute and try again."
