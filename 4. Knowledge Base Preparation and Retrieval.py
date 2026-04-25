# 2. RAG / Knowledge Base Prep & Retrieval
def load_and_prepare_knowledge_base(kb_path):
    """
    Loads the JSON. Because the original JSON contains Google embeddings,
    we must re-embed the text chunks using our local MiniLM model.
    """
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"Knowledge base file '{kb_path}' not found.")

    with open(kb_path, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)

    print("Preparing local embeddings for the knowledge base...")
    local_kb =[]

    # Re-embed using the local model
    for item in knowledge_base:
        chunk_text = item.get("text", "")
        if chunk_text:
            # Generate local 384-dimensional embedding
            local_embedding = embedder.encode(chunk_text, convert_to_tensor=True)
            local_kb.append({
                "text": chunk_text,
                "embedding": local_embedding
            })

    return local_kb

def retrieve_context(query, local_kb, top_k=3):
    """Embeds the query and finds the top_k most similar chunks locally."""
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    scored_chunks =[]
    for item in local_kb:
        # Compute cosine similarity using sentence-transformers util
        score = util.cos_sim(query_embedding, item["embedding"]).item()
        scored_chunks.append((score, item["text"]))

    # Sort by descending cosine similarity score
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # Return the text of the top K highest-scoring chunks
    return [chunk[1] for chunk in scored_chunks[:top_k]]