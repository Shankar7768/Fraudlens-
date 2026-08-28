import torch
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "FacebookAI/roberta-base"


class RoBERTaEmbedder:
    def __init__(self):
        print("Loading RoBERTa model...")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)

        self.model.eval()

        print("RoBERTa loaded successfully.")

    def encode(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use the CLS representation as the job-posting embedding
        embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings


if __name__ == "__main__":
    embedder = RoBERTaEmbedder()

    sample_text = [
        "Software Engineer. Looking for a Python developer with experience in SQL."
    ]

    embeddings = embedder.encode(sample_text)

    print("Embedding shape:", embeddings.shape)