# Main Flow
def main():
    print("=== Offline Edge RAG System ===\n")

    # Step 1: Load and compute local embeddings (done once in memory)
    try:
        local_kb = load_and_prepare_knowledge_base(KNOWLEDGE_BASE_FILE)
    except Exception as e:
        print(f"Failed to load knowledge base: {e}")
        return

    while True:
        # Step 2: Get Query
        query = get_user_input()
        if not query:
            print("Empty query provided. Exiting.")
            break

        print("\n[1/2] Searching local knowledge base...")
        relevant_contexts = retrieve_context(query, local_kb, top_k=3)

        print("[2/2] Generating answer using local LLM...\n")
        answer = generate_answer(query, relevant_contexts)

        print("================== ANSWER ==================")
        print(answer)
        print("============================================\n")

        cont = input("Ask another question? (y/n): ").strip().lower()
        if cont != 'y':
            break

if __name__ == "__main__":
    main()