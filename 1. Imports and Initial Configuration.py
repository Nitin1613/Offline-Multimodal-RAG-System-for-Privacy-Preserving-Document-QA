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
