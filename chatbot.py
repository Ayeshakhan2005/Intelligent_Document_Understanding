import re
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"={3,}.*?={3,}", " ", text, flags=re.DOTALL)
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_into_chunks(text, chunk_size=80):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def search_document(question, text):

    if not text or not text.strip():
        return "No document text is available."

    question = question.strip()

    if not question:
        return "Please enter a question."

    text = clean_text(text)

    chunks = split_into_chunks(text)

    if not chunks:
        return "I could not find information in the document."

    # Create embeddings
    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    chunk_embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    # Cosine similarity
    scores = np.dot(
        chunk_embeddings,
        question_embedding
    )

    # Get best chunks
    best_indices = np.argsort(scores)[::-1][:3]

    best_chunks = [
        chunks[i]
        for i in best_indices
        if scores[i] > 0.25
    ]

    if not best_chunks:
        return "I could not find the answer in the document."

    # Return the most relevant context
    return "\n\n".join(best_chunks[:2])
