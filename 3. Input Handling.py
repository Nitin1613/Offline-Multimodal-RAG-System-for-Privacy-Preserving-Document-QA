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

# VOICE INPUT

else:

    st.write("### 🎤 Voice Query")

    audio = mic_recorder(
        start_prompt="🎙 Start Recording",
        stop_prompt="⏹ Stop Recording",
        just_once=True,
        use_container_width=True
    )

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

    st.session_state.voice_query = st.text_area(
        "Transcribed Query",
        value=st.session_state.voice_query,
        height=120
    )

    query = st.session_state.voice_query
