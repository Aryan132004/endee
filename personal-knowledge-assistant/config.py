import os
from dotenv import load_dotenv
load_dotenv()

LLM_PROVIDER   = os.getenv("LLM_PROVIDER",   "gemini")
LLM_MODEL      = os.getenv("LLM_MODEL",      "gemini-1.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY",  "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",    "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY",  "")

ENDEE_BASE_URL   = os.getenv("ENDEE_BASE_URL",   "http://localhost:8080/api/v1")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN",  "")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM   = int(os.getenv("EMBEDDING_DIM", "384"))

INDEX_NAME    = os.getenv("INDEX_NAME",    "knowledge_base")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K         = int(os.getenv("TOP_K",         "5"))
