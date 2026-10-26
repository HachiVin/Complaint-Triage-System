"""Deploy the API and dashboard on Modal, with IndoBERT served live.

One-time setup (Windows PowerShell, in the project venv):

    pip install modal
    python -m modal setup                     # opens the browser to log in

In the Modal dashboard: Secrets -> Create new secret -> Custom, named "complaint-triage", with
DATABASE_URL (the Neon connection string) and, optionally, LLM_API_KEY for the agent.

Deploy (and again after every change):

    python -m modal deploy modal_app.py

Cost control on the free plan: the container starts on the first request, stops two minutes after
the last one, and never runs more than one copy at a time. The model is baked into the image, so a
cold start does not download it again.
"""

import modal

HF_REPO = "HachiVin/complaint-triage-indobert"


def download_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(HF_REPO)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("torch==2.14.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("transformers==5.20.2", "tokenizers==0.23.3", "huggingface_hub==2.3.0")
    .run_function(download_model)
    .env({
        "MODEL_DIR": f"hf://{HF_REPO}",
        "HF_HUB_OFFLINE": "1",  # use the copy baked into the image, no download at start-up
        "ALLOW_CORRECTIONS": "0",  # public demo: visitors cannot change stored labels
        "PYTHONPATH": "/root",
    })
    .add_local_dir("complaint_triage", "/root/complaint_triage", ignore=["**/__pycache__"])
    .add_local_dir("app", "/root/app", ignore=["**/__pycache__"])
    .add_local_dir("sql", "/root/sql")
)

app = modal.App("complaint-triage", image=image)


@app.function(
    secrets=[modal.Secret.from_name("complaint-triage")],
    cpu=1.0,
    memory=2048,
    scaledown_window=120,
    max_containers=1,
    timeout=120,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    from app.main import create_app

    return create_app()