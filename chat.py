from src.embeddings.generator import LocalEmbedder
from src.storage.vector_store import ChromaVectorStore
from src.generation.llm import GeminiLLM, LocalLLM

def start_chat():
    print("=== STARTING NIT PATNA ASSISTANT ===")
    print("Loading database and models... (This is fast because we are NOT extracting PDFs!)")
    
    embedder = LocalEmbedder()
    db = ChromaVectorStore()
    # llm = LocalLLM(model_name="llama3")
    llm = GeminiLLM() 
    
    print("\nSystem ready! Type 'exit' or 'quit' to stop.")
    
    while True:
        user_query = input("\nStudent: ")
        
        if user_query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not user_query.strip():
            continue
            
        # 1. Embed the query (Fast)
        query_vector = embedder.embed_text(user_query)
        
        # 2. Retrieve relevant chunks from ChromaDB (Fast)
        # results = db.search(query_embedding=query_vector, n_results=2)
        results = db.search(query_embedding=query_vector, n_results=10)
        
        if not results['documents'] or len(results['documents'][0]) == 0:
            print("AI: I couldn't find any relevant information in the database.")
            continue
            
        # 3. Prepare context and deterministic sources
        retrieved_text = "\n\n".join(results['documents'][0])
        sources = set()
        for metadata in results['metadatas'][0]:
            sources.add(metadata.get('source', 'Unknown Document'))
            
        # 4. Generate answer via LLM
        print("AI: Thinking...")
        final_answer = llm.generate_answer(query=user_query, context=retrieved_text)
        
        # 5. Output with Citations
        print("\n--- Answer ---")
        print(final_answer)
        
        if "I don't know" not in final_answer:
            print("\n[Sources Used:]")
            for source in sources:
                print(f"- {source}")
        print("--------------")

if __name__ == "__main__":
    start_chat()

