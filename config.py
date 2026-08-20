"""Configuration for RAG Chatbot"""

import os


# =====================================================
# PROVIDER
# =====================================================

PROVIDER = os.getenv(
    "PROVIDER",
    "gemini",
).lower()


# =====================================================
# EMBEDDINGS
# =====================================================

# We use local embeddings so Gemini is only used
# for generating the final answer.
EMBEDDINGS_PROVIDER = os.getenv(
    "EMBEDDINGS_PROVIDER",
    "local",
).lower()

LOCAL_EMBEDDINGS_MODEL = os.getenv(
    "LOCAL_EMBEDDINGS_MODEL",
    "all-MiniLM-L6-v2",
)


# =====================================================
# OPENAI
# =====================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "",
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini",
)


# =====================================================
# GEMINI
# =====================================================

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY",
    "",
)

# IMPORTANT:
# Use the Gemini model available to your API account.
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)

GEMINI_EMBEDDINGS_MODEL = os.getenv(
    "GEMINI_EMBEDDINGS_MODEL",
    "models/embedding-001",
)


# =====================================================
# MODEL
# =====================================================

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    GEMINI_MODEL,
)

TEMPERATURE = float(
    os.getenv(
        "TEMPERATURE",
        "0.3",
    )
)


# =====================================================
# RAG
# =====================================================

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "1000",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "200",
    )
)

K_DOCUMENTS = int(
    os.getenv(
        "K_DOCUMENTS",
        "5",
    )
)


# =====================================================
# RETRIEVAL SAFETY
# =====================================================

# Lower threshold works better with the current
# local MiniLM + Chroma distance conversion.

RETRIEVAL_THRESHOLD = float(
    os.getenv(
        "RETRIEVAL_THRESHOLD",
        "0.20",
    )
)


# =====================================================
# MANIFEST
# =====================================================

MANIFEST_PATH = os.getenv(
    "MANIFEST_PATH"
)


# =====================================================
# LOGGING
# =====================================================

LOG_PATH = os.getenv(
    "LOG_PATH",
    "logs/qa.jsonl",
)

AUDIT_DB_PATH = os.getenv(
    "AUDIT_DB_PATH",
    "logs/audit.db",
)


# =====================================================
# STREAMLIT
# =====================================================

PAGE_TITLE = os.getenv(
    "PAGE_TITLE",
    "RAG Chatbot",
)

PAGE_ICON = os.getenv(
    "PAGE_ICON",
    "🤖",
)