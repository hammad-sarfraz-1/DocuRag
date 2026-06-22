# Python 3.11: matches the tested dependency stack (3.13 pulled untested wheels).
FROM python:3.11-slim

WORKDIR /app

# Prevent Python buffering; keep the embedding/reranker caches in a known path.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/.cache/huggingface

# System libraries for OpenCV / EasyOCR / pdf2image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install pinned dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding + reranker models into the image so the container
# starts fast and works without reaching Hugging Face at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy app
COPY backend/ backend/
COPY frontend/ frontend/
COPY run.py .

# Persist the vector store across restarts by mounting storage at /app/chroma_db
# (Docker: `-v docurag_data:/app/chroma_db`; Railway: add a Volume with mount path
# /app/chroma_db). No VOLUME instruction — Railway rejects it.
EXPOSE 8000

# Longer start period: first boot loads the ML models from disk before /health is ready.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# GROQ_API_KEY is required at runtime, e.g.:
#   docker run -p 8000:8000 -e GROQ_API_KEY=gsk_... -v docurag_data:/app/chroma_db docurag
# Shell form (not exec form) so ${PORT} expands. Railway injects $PORT at runtime
# and routes the public domain to it; locally we fall back to 8000.
CMD uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}
