import os
import numpy as np
import torch

from training.database import load_job_postings
from training.preprocess import preprocess_data
from training.roberta_embeddings import RoBERTaEmbedder


OUTPUT_PATH = "data/features/roberta_embeddings.npy"


def generate_embeddings():
    df, _ = preprocess_data()

    texts = df["job_text"].tolist()

    embedder = RoBERTaEmbedder()

    all_embeddings = []

    batch_size = 16

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        embeddings = embedder.encode(batch)

        all_embeddings.append(
            embeddings.cpu().numpy()
        )

        print(
            f"Processed {min(start + batch_size, len(texts))}"
            f"/{len(texts)} job postings"
        )

    embeddings = np.vstack(all_embeddings)

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    np.save(OUTPUT_PATH, embeddings)

    print("\nRoBERTa embedding generation completed.")
    print("Embedding shape:", embeddings.shape)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    generate_embeddings()
