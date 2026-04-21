!pip install speechrecognition openai-whisper
import os
import json
import warnings
import numpy as np
import speech_recognition as sr
import whisper
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# Configuration & Global Variables
KNOWLEDGE_BASE_FILE = "my_pdf_embeddings.json"