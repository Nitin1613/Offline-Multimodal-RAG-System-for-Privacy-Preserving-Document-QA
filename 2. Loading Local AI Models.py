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