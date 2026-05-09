import os
import glob
import warnings
import tempfile
import streamlit as st
import speech_recognition as sr
import whisper
import PyPDF2

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# CONFIG

PDF_DIRECTORY = "my_pdfs"
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Offline Edge RAG", layout="wide")

st.title("📚 Offline Edge RAG System")

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

# CHUNKING

def get_text_chunks(text, chunk_size=1000, overlap=100):

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size
        chunks.append(text[start:end])

        start += (chunk_size - overlap)

    return chunks

# PDF EXTRACTION

def extract_text_from_pdfs(directory_path):

    all_chunks = []

    pdf_files = glob.glob(
        os.path.join(directory_path, "*.pdf")
    )

    if not pdf_files:
        st.error(f"No PDFs found in '{directory_path}'")
        return []

    for pdf_file in pdf_files:

        st.write(f"Loading: {os.path.basename(pdf_file)}")

        text = ""

        try:
            with open(pdf_file, "rb") as f:

                reader = PyPDF2.PdfReader(f)

                for page in reader.pages:

                    extracted = page.extract_text()

                    if extracted:
                        text += extracted + "\n"

            pdf_chunks = get_text_chunks(text)

            all_chunks.extend(pdf_chunks)

        except Exception as e:
            st.error(f"Error reading {pdf_file}: {e}")

    return all_chunks

# KNOWLEDGE BASE

@st.cache_resource
def prepare_knowledge_base():

    chunks = extract_text_from_pdfs(PDF_DIRECTORY)

    local_kb = []

    for chunk in chunks:

        if chunk.strip():

            embedding = embedder.encode(
                chunk,
                convert_to_tensor=True
            )

            local_kb.append({
                "text": chunk,
                "embedding": embedding
            })

    return local_kb

# RETRIEVAL

def retrieve_context(query, local_kb, top_k=3):

    query_embedding = embedder.encode(
        query,
        convert_to_tensor=True
    )

    scored_chunks = []

    for item in local_kb:

        score = util.cos_sim(
            query_embedding,
            item["embedding"]
        ).item()

        scored_chunks.append(
            (score, item["text"])
        )

    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [x[1] for x in scored_chunks[:top_k]]

# ANSWER GENERATION

def generate_answer(query, context_chunks):

    context_text = "\n\n---\n\n".join(context_chunks)

    messages = [
        {
            "role": "system",
            "content":
            "Answer ONLY using the provided context."
        },
        {
            "role": "user",
            "content":
            f"Context:\n{context_text}\n\nQuestion:\n{query}"
        }
    ]

    prompt = llm_pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    outputs = llm_pipeline(
        prompt,
        max_new_tokens=256,
        temperature=0.3,
        do_sample=True
    )

    generated_text = outputs[0]["generated_text"]

    answer = generated_text[len(prompt):].strip()

    return answer

# PREPARE KB

with st.spinner("Preparing Knowledge Base..."):
    local_kb = prepare_knowledge_base()

st.success(f"Knowledge Base Ready ({len(local_kb)} chunks)")

# SESSION STATE

if "recording" not in st.session_state:
    st.session_state.recording = False

if "audio_data" not in st.session_state:
    st.session_state.audio_data = None

if "voice_query" not in st.session_state:
    st.session_state.voice_query = ""


# INPUT METHOD

method = st.radio(
    "Choose Input Method",
    ["Text", "Voice"]
)

query = ""

# TEXT INPUT

if method == "Text":

    query = st.text_input(
        "Enter your question"
    )

    if st.button("Answer Query"):

        if not query:
            st.warning("Please enter a question")

        else:

            with st.spinner("Searching Knowledge Base..."):

                contexts = retrieve_context(
                    query,
                    local_kb
                )

            with st.spinner("Generating Answer..."):

                answer = generate_answer(
                    query,
                    contexts
                )

            st.subheader("Answer")
            st.write(answer)
 

# VOICE INPUT

else:

    from streamlit_mic_recorder import mic_recorder

    st.write("### 🎤 Voice Query")

    audio = mic_recorder(
        start_prompt="🎙 Start Recording",
        stop_prompt="⏹ Stop Recording",
        just_once=True,
        use_container_width=True
    )

    # TRANSCRIBE


    if audio:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_audio:

            temp_audio.write(audio["bytes"])

            temp_path = temp_audio.name

        with st.spinner("Transcribing Audio..."):

            result = asr_model.transcribe(
                temp_path
            )

            st.session_state.voice_query = (
                result["text"].strip()
            )

        os.remove(temp_path)

        st.success("Transcription Completed")

    # SHOW TRANSCRIBED TEXT

    if "voice_query" not in st.session_state:
        st.session_state.voice_query = ""

    st.session_state.voice_query = st.text_area(
        "Transcribed Query",
        value=st.session_state.voice_query,
        height=120
    )
    
    # ANSWER BUTTON

    if st.button("Answer Query"):

        query = st.session_state.voice_query

        if not query:

            st.warning("No query found")

        else:

            with st.spinner(
                "Searching Knowledge Base..."
            ):

                contexts = retrieve_context(
                    query,
                    local_kb
                )

            with st.spinner(
                "Generating Answer..."
            ):

                answer = generate_answer(
                    query,
                    contexts
                )

            st.subheader("Answer")
            st.write(answer)

            with st.expander(
                "Retrieved Context"
            ):

                for i, ctx in enumerate(
                    contexts,
                    1
                ):

                    st.markdown(
                        f"### Chunk {i}"
                    )

                    st.write(ctx)
