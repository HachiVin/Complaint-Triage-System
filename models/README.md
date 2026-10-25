# Models

- `baseline/`: TF-IDF + logistic regression. The copy in the repo is the **demo** model
  (`demo-synthetic`), trained on fictional template reviews by `scripts/make_demo_model.py`.
  Replace it with one trained on the real dataset (`scripts/train_baseline.py`).
- `indobert-multitask/`: not committed (about 500 MB). Train it with `notebooks/train_colab.ipynb`,
  then publish it as a GitHub release asset or on the Hugging Face Hub and point `MODEL_DIR` at it.
