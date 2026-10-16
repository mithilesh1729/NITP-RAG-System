from src.embeddings.generator import GeminiEmbedder
from src.storage.vector_store import ChromaVectorStore
from src.generation.llm import GeminiLLM

def start_chat():
    print("=== STARTING NIT PATNA ASSISTANT ===")
    print("Loading database and models...")
    
    embedder = GeminiEmbedder()
    db = ChromaVectorStore()
    llm = GeminiLLM()
    
    print("\nSystem ready! Type 'exit' or 'quit' to stop.")
    
    while True:
        user_query = input("\nStudent: ")
        
        if user_query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not user_query.strip():
            continue
            
        # 1. Embed the query using Gemini
        query_vector = embedder.embed_text(user_query)
        
        # 2. Retrieve relevant chunks using HYBRID SEARCH (M.Tech feature)
        # We can safely retrieve 25 chunks because Gemini Flash has a 1 MILLION token context window!
        # This solves table isolation and sub-table context vacuum issues.
        results = db.hybrid_search(query_text=user_query, query_embedding=query_vector, n_results=25)
        
        if not results['documents'] or len(results['documents'][0]) == 0:
            print("AI: I couldn't find any relevant information in the database.")
            continue
            
        # 3. Prepare context and deterministic sources
        retrieved_text = "\n\n".join(results['documents'][0])
        sources = set()
        for metadata in results['metadatas'][0]:
            sources.add(metadata.get('source', 'Unknown Document'))
            
        if 'hybrid_scores' in results:
            print("\n[Hybrid Search Scoring]:")
            for score_info in results['hybrid_scores'][0]:
                print(f"  - {score_info}")
            
        # 4. Generate answer via LLM
        print("AI: Thinking...")
        final_answer = llm.generate_answer(query=user_query, context=retrieved_text)
        
        # 5. Output with Citations
        print("\n--- Answer ---")
        print(final_answer)
        
        is_refusal = final_answer.strip().startswith("I don't know") and len(final_answer.strip()) < 80
        if not is_refusal:
            print("\n[Sources Used:]")
            for source in sources:
                print(f"- {source}")
        print("--------------")

if __name__ == "__main__":
    start_chat()
