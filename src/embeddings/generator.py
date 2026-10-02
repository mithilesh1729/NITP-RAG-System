import os

from abc import ABC, abstractmethod
from typing import List
from sentence_transformers import SentenceTransformer

class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding generators.
    This guarantees that any embedder we use in the future 
    (OpenAI, Local, etc.) will have the exact same method signature.
    """
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Converts a single string into a vector."""
        pass
        
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Converts a list of strings into a list of vectors."""
        pass

class LocalEmbedder(BaseEmbedder):
    """
    Generates embeddings locally using HuggingFace's sentence-transformers.
    100% free, runs on CPU/GPU without API keys.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading local embedding model: {model_name}...")
        # This will download the model the very first time it is run (~80MB)
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully.")
        
    def embed_text(self, text: str) -> List[float]:
        # encode() returns a numpy array. We convert it to a standard python list.
        vector = self.model.encode(text)
        return vector.tolist()
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # encode() can handle a list of strings efficiently in parallel
        vectors = self.model.encode(texts)
        return vectors.tolist()

