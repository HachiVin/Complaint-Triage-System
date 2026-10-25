# Hugging Face Space (Docker SDK). Builds the app straight from the GitHub repository, so the Space repo
# only needs this file and README.md. Settings -> Variables and secrets:
#   variables: MODEL_DIR=hf://<hf-username>/complaint-triage-indobert   ALLOW_CORRECTIONS=0
#   secrets:   DATABASE_URL (Neon connection string)   LLM_API_KEY (optional, enables "Tanya data")
# To pick up new commits on GitHub: Settings -> Factory rebuild.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

ARG REPO=https://github.com/HachiVin/Complaint-Triage-System.git
ARG REF=main
RUN git clone --depth 1 --branch ${REF} ${REPO} .

RUN pip install --user -r requirements.txt \
 && pip install --user --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
 && pip install --user transformers==5.20.2 tokenizers==0.23.3 huggingface_hub==2.3.0

ENV PORT=7860 MODEL_DIR=models/baseline
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]