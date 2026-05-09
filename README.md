# 📚 Offline Edge RAG System

An end-to-end **Offline Retrieval-Augmented Generation (RAG)** application built using:

- Streamlit
- Whisper ASR
- Sentence Transformers
- Local LLM (Qwen2.5)
- PDF-based Knowledge Base

The system allows users to ask questions using either:

- ✍️ Text Queries
- 🎤 Voice Queries

and generates answers strictly from locally stored PDF documents without requiring external APIs.

---

# 🚀 Features

## ✅ Fully Offline AI Pipeline

Runs completely on local machine:

- Local Embedding Model
- Local LLM
- Local Speech-to-Text
- Local Vector Retrieval

No OpenAI API required.

---

## ✅ PDF-Based Knowledge Base

- Automatically scans all PDFs inside `my_pdfs/`
- Extracts text from PDFs
- Splits text into semantic chunks
- Creates vector embeddings for retrieval

---

## ✅ Retrieval-Augmented Generation (RAG)

Implements semantic search pipeline:

1. Embed user query
2. Retrieve top-k relevant chunks
3. Inject context into prompt
4. Generate grounded answer using local LLM

---

## ✅ Voice Query Support

- Browser microphone recording
- Start/Stop recording buttons
- Whisper-based transcription
- Automatic answer generation from transcribed query

---

# 🧠 Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Speech Recognition | Whisper |
| Embedding Model | nomic-embed-text-v1.5 |
| LLM | Qwen2.5-0.5B-Instruct |
| Vector Similarity | Sentence Transformers |
| PDF Processing | PyPDF2 |

---
 
