import os
from abc import ABC, abstractmethod
from typing import List

class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding generators.
    """
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass
        
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass

class LocalEmbedder(BaseEmbedder):
    """
    Generates embeddings locally using HuggingFace's sentence-transformers.
    Has a strict 256 token (~1000 char) limit. Will silently truncate long text!
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"Loading local embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully.")
        
    def embed_text(self, text: str) -> List[float]:
        vector = self.model.encode(text)
        return vector.tolist()
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts)
        return vectors.tolist()

class GeminiEmbedder(BaseEmbedder):
    """
    Generates embeddings using Google's text-embedding-004 API.
    Supports up to 2048 tokens (~8000 characters). 
    Perfect for keeping massive markdown tables intact without truncation!
    """
    def __init__(self):
        from google import genai
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found! Check your .env file.")
            
        self.client = genai.Client(api_key=api_key, http_options={'client_args': {'verify': False}})
        self.model = 'gemini-embedding-2'
        print(f"Loaded Cloud Embedding Model: {self.model} (large token limit)")

    def embed_text(self, text: str) -> List[float]:
        # Google API requires text to be non-empty
        if not text.strip():
            return [0.0] * 768
            
        import time
        from google.genai import errors
        
        # Free tier rate limits exist, so we need a retry loop
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=text,
                )
                return result.embeddings[0].values
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_retries - 1:
                        print(f"    [Rate Limited] Waiting 30 seconds before retrying...")
                        time.sleep(30)
                        continue
                raise e

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            vectors.append(self.embed_text(text))
        return vectors
