Colab / Kaggle workflow (no local GPU):
1. Develop and test code locally (CPU).
2. In Colab/Kaggle: `!git clone <your-repo>` then `!pip install -e ".[ml]"`.
3. Run heavy steps (embeddings, fine-tuning) there, save outputs to Drive / Kaggle datasets.
