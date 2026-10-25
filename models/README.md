# Models

- `baseline/`: TF-IDF + logistic regression, version `v1`, trained on the USH BRImo training split
  (36,470 reviews) by `scripts/train_baseline.py`. Small enough to commit (34 MB) and served by default.
  `scripts/make_demo_model.py` rebuilds the old demo model from fictional reviews if you need it.
- `indobert-mt/`: not committed (about 500 MB). Train it with `notebooks/train_colab.ipynb`, then
  publish it as a GitHub release asset or on the Hugging Face Hub and point `MODEL_DIR` at it.