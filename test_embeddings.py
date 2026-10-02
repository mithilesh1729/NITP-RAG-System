from src.embeddings.generator import LocalEmbedder

def test_local_embeddings():
    print("1. Initializing the Local Embedder (this may download files on first run)...")
    # Initialize our wrapper class
    embedder = LocalEmbedder()
    
    print("\n2. Generating an embedding for a test sentence...")
    test_sentence = "What is the fee structure for the Dual Degree program at NIT Patna?"
    print(f"Sentence: '{test_sentence}'")
    
    # Generate the vector
    vector = embedder.embed_text(test_sentence)
    
    print(f"\n3. Results:")
    print(f"Vector dimensions (length): {len(vector)}")
    print("First 5 numbers in the vector:")
    
    # Print just the first 5 numbers to verify it is a list of floats
    print([round(num, 4) for num in vector[:5]])
    
    print("\nSuccess! The text has been mathematically mapped into a semantic space.")

if __name__ == "__main__":
    test_local_embeddings()

