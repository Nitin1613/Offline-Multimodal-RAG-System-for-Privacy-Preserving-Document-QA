# TEXT CHUNKING

def get_text_chunks(
    text,
    chunk_size=1000,
    overlap=100
):

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

        st.error(
            f"No PDFs found in '{directory_path}'"
        )

        return []

    for pdf_file in pdf_files:

        st.write(
            f"Loading: {os.path.basename(pdf_file)}"
        )

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

            st.error(
                f"Error reading {pdf_file}: {e}"
            )

    return all_chunks

# KNOWLEDGE BASE CREATION

@st.cache_resource
def prepare_knowledge_base():

    chunks = extract_text_from_pdfs(
        PDF_DIRECTORY
    )

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

def retrieve_context(
    query,
    local_kb,
    top_k=3
):

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

    return [
        x[1]
        for x in scored_chunks[:top_k]
    ]


with st.spinner("Preparing Knowledge Base..."):

    local_kb = prepare_knowledge_base()

st.success(
    f"Knowledge Base Ready ({len(local_kb)} chunks)"
)
