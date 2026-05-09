import os
import glob
import warnings
import tempfile

import streamlit as st
import whisper
import PyPDF2

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from streamlit_mic_recorder import mic_recorder

# CONFIGURATION

PDF_DIRECTORY = "my_pdfs"

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Offline Edge RAG",
    layout="wide"
)

st.title("📚 Offline Edge RAG System")

# SESSION STATE

if "voice_query" not in st.session_state:
    st.session_state.voice_query = ""
