@st.cache_resource
def load_models():

    # Embedding Model
    embedder = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True
    )

    # Local LLM
    llm_pipeline = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        device_map="auto"
    )

    # Whisper ASR
    asr_model = whisper.load_model("base")

    return embedder, llm_pipeline, asr_model


with st.spinner("Loading AI Models..."):

    embedder, llm_pipeline, asr_model = load_models()

st.success("Models Loaded Successfully")
