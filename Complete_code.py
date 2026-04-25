!pip install speechrecognition openai-whisper
import os
import json
import warnings
import numpy as np
import speech_recognition as sr
import whisper
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# Configuration & Global Variables
KNOWLEDGE_BASE_FILE = "my_pdf_embeddings.json"

print("Loading local AI models... (This may take a moment and requires internet on the FIRST run)")

# 1. Load Local Embedding Model
# all-MiniLM-L6-v2 is ultra-fast and uses ~100MB of RAM
print("[1/3] Loading Embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Load Local LLM (Qwen2.5-0.5B-Instruct)
# Ultra-lightweight LLM,runs well on edge CPU
print("[2/3] Loading LLM...")
llm_pipeline = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device_map="auto" #auto-select GPU if available, otherwise runs on CPU
)

# 3. Load Local ASR (Whisper Base)
print("[3/3] Loading Whisper Voice Model")
asr_model = whisper.load_model("base") # ~140MB model

print("All models loaded successfully!\n")

# 1. Input Handling (Offline Voice or Text)
def get_user_input():
    print("Choose input method:")
    print("1. Text")
    print("2. Voice (Local Whisper)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '2':
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("\nListening... Please speak your query.")
            recognizer.adjust_for_ambient_noise(source)
            try:
                # Capture the audio
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)

                # Save temporarily to process with local Whisper
                temp_wav = "temp_audio.wav"
                with open(temp_wav, "wb") as f:
                    f.write(audio.get_wav_data())

                print("Transcribing locally...")
                result = asr_model.transcribe(temp_wav)
                text = result["text"].strip()

                # Cleanup
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)

                print(f"Transcribed Text: '{text}'\n")
                return text

            except sr.WaitTimeoutError:
                print("Listening timed out. Falling back to text input.")
            except Exception as e:
                print(f"Voice input error: {e}. Falling back to text input.")

    # Default to text input
    return input("\nEnter your text query: ").strip()

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

# 3. Prompt Augmentation & Local LLM Summarization
def generate_answer(query, context_chunks):
    """Combines query and context, then queries the local Qwen LLM."""

    context_text = "\n\n---\n\n".join(context_chunks)

    # Format messages using the standard Chat format
    messages =[
        {
            "role": "system",
            "content": "You are a highly capable, local AI assistant. Please answer the user's query based ONLY on the provided context. If the context does not contain the answer, say 'I cannot find the answer in the provided context.'"
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nUser Query:\n{query}"
        }
    ]

    # Apply the model's specific chat template
    prompt = llm_pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate the output
    outputs = llm_pipeline(
        prompt,
        max_new_tokens=256,
        temperature=0.3, # Low temperature for factual RAG responses
        do_sample=True
    )

    # Extract only the generated answer (strip the prompt)
    generated_text = outputs[0]["generated_text"]
    answer = generated_text[len(prompt):].strip()

    return answer

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