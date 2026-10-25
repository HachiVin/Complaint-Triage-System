# API image. Build arg WITH_TORCH=1 adds CPU-only torch + transformers for the IndoBERT model;
# without it the image serves the TF-IDF baseline and stays small.
FROM python:3.12-slim

ARG WITH_TORCH=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt \
 && if [ "$WITH_TORCH" = "1" ]; then \
      pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
      && pip install transformers==5.20.2 tokenizers==0.23.3; \
    fi

COPY complaint_triage ./complaint_triage
COPY app ./app
COPY sql ./sql
COPY models ./models
COPY data/sample ./data/sample

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

ENV MODEL_DIR=models/baseline PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8000\")}/health')" || exit 1
# Cloud Run sets $PORT; one worker keeps a single model copy in memory.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
