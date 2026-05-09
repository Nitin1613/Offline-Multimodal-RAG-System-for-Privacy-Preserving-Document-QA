# LOAD MODELS
@st.cache_resource
def load_models():

    embedder = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True
    )

    llm_pipeline = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        device_map="auto"
    )

    asr_model = whisper.load_model("base")

    return embedder, llm_pipeline, asr_model

with st.spinner("Loading Models..."):
    embedder, llm_pipeline, asr_model = load_models()

st.success("Models Loaded Successfully")
