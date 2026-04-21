## Local AI: Offline Edge RAG System

This project implements a fully **Offline Edge RAG (Retrieval-Augmented Generation) System** designed to run on local hardware without an internet connection. It allows users to interact with their own documents (PDFs) via text or voice while maintaining total data privacy.

### Core Architecture
The system utilizes a modular "Edge" stack to ensure performance on standard CPUs:
* **Voice Processing**: Uses **OpenAI Whisper (Base)** for local Automatic Speech Recognition (ASR), converting spoken queries into text without cloud APIs.
* **Embedding Engine**: Employs `all-MiniLM-L6-v2` to transform document chunks into 384-dimensional vectors.
* **Local LLM**: Features the **Qwen2.5-0.5B-Instruct** model, an ultra-lightweight LLM optimized for fast text generation on edge devices.



### How it Works
1.  **Knowledge Preparation**: The system loads a JSON file containing text from PDFs. It re-indexes this content using local embeddings to ensure compatibility with the offline environment.
2.  **Hybrid Input**: Users choose between typing their query or speaking. If voice is selected, Whisper transcribes the audio locally.
3.  **Contextual Retrieval**: The system performs a semantic search using cosine similarity to find the most relevant information within the knowledge base.
4.  **Grounded Generation**: The retrieved context is fed into the Qwen LLM. The model is strictly instructed to answer based **only** on the provided context to prevent hallucinations.

### Key Benefits
* **Privacy First**: No data ever leaves the local machine.
* **Efficiency**: Optimized for "edge" use, meaning it runs smoothly without high-end GPU requirements.
* **Zero Latency**: Eliminates the need for API calls or internet stability after the initial setup.
