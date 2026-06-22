import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for cosine similarity. Install it with `pip install scikit-learn`."
    ) from exc

CHUNKS_PATH = Path(__file__).parent / "transcript_chunks" / "chunks.json"
EMBEDDINGS_MODEL = "bge-m3"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
OUTPUT_BASE = Path(__file__).parent / "chunks_with_embeddings"
CSV_FILE = OUTPUT_BASE.with_suffix(".csv")
PARQUET_FILE = OUTPUT_BASE.with_suffix(".parquet")

# Keep the retrieved context window very small to avoid prompt overflow.
MAX_CONTEXT_CHUNKS = 1
MAX_CHUNK_CHARACTERS = 800

SYSTEM_PROMPT = (
    "You are a retrieval-augmented AI assistant. Use only the retrieved transcript chunks "
    "to answer the user question. Do not invent information or hallucinate. "
    "If the answer is not present in the provided chunks, respond with: "
    "'I don\'t know based on the provided sources.'"
)

def truncate_chunk_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def build_rag_prompt(query: str, retrieved_df: pd.DataFrame) -> str:
    chunk_entries = []
    for _, row in retrieved_df.iterrows():
        source = f"[{row.get('tutorial_number', '')}] {row.get('tutorial_name', '')} chunk {row.get('chunk_id', '')}"
        text = truncate_chunk_text(row.get('text', '').strip(), MAX_CHUNK_CHARACTERS)
        chunk_entries.append(f"{source}\n{text}")

    retrieved_text = "\n\n".join(chunk_entries)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Retrieved chunks:\n"
        f"{retrieved_text}\n\n"
        f"User question: {query}\n\n"
        "Answer:"
    )

print(f"Loading chunks from: {CHUNKS_PATH}")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks_data = json.load(f)

if not chunks_data:
    raise ValueError("chunks.json is empty")

# Create DataFrame from chunks
df = pd.DataFrame(chunks_data)
print(f"Initial DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

if "text" not in df.columns:
    raise ValueError("Expected column 'text' not found in chunks.json")


def load_saved_dataframe() -> pd.DataFrame | None:
    if CSV_FILE.exists():
        print(f"Loading saved embeddings from: {CSV_FILE}")
        return pd.read_csv(CSV_FILE)
    if PARQUET_FILE.exists():
        print(f"Loading saved embeddings from: {PARQUET_FILE}")
        try:
            return pd.read_parquet(PARQUET_FILE)
        except (ImportError, ValueError) as e:
            print(f"Failed to load parquet file: {e}")
            return None
    return None


def embedding_columns(dataframe: pd.DataFrame) -> list[str]:
    return [col for col in dataframe.columns if col.startswith("embedding_")]


def request_embedding(text: str) -> np.ndarray:
    response = requests.post(
        OLLAMA_URL,
        json={"prompt": text, "model": EMBEDDINGS_MODEL},
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Embedding request failed with status {response.status_code}: {response.text}"
        )

    data = response.json()
    embedding = data.get("embeddings") or data.get("embedding") or []

    if not isinstance(embedding, list):
        raise ValueError(f"Unexpected embedding type: {type(embedding)}")
    if not embedding:
        raise ValueError("Received empty embedding")

    return np.array(embedding, dtype=float)


saved_df = load_saved_dataframe()
if saved_df is not None and embedding_columns(saved_df):
    # Only reuse saved embeddings if the saved chunk identifiers match the current chunks.
    if set(saved_df.get("chunk_file", [])) == set(df.get("chunk_file", [])) and len(saved_df) == len(df):
        print("Found saved embeddings for all current chunks. Reusing saved file.")
        df = saved_df.copy()
    else:
        print("Saved embeddings file exists but does not match current chunks. Merging known embeddings and computing missing ones.")
        emb_cols = embedding_columns(saved_df)
        merged = df.merge(
            saved_df[["chunk_file"] + emb_cols].drop_duplicates("chunk_file"),
            on="chunk_file",
            how="left",
        )
        df = merged
else:
    print("No saved embeddings found. Generating embeddings for all chunks...")

# Identify missing embeddings
emb_cols = embedding_columns(df)
if emb_cols:
    missing_mask = df[emb_cols].isna().any(axis=1)
else:
    missing_mask = pd.Series([True] * len(df), index=df.index)

if missing_mask.any():
    print(f"Computing embeddings for {missing_mask.sum()} missing chunks...")
    new_embeddings = []
    for idx, row in df[missing_mask].iterrows():
        text = row["text"]
        embedding = request_embedding(text)
        new_embeddings.append((idx, embedding))
        print(f"Chunk {idx + 1}/{len(df)} - embedding length: {embedding.shape[0]}")

    if not emb_cols:
        embedding_dim = new_embeddings[0][1].shape[0]
        emb_cols = [f"embedding_{i}" for i in range(embedding_dim)]
        for col in emb_cols:
            df[col] = np.nan

    for idx, embedding in new_embeddings:
        for dim, col in enumerate(emb_cols):
            df.at[idx, col] = embedding[dim]
else:
    print("No missing embeddings found; using cached embeddings.")

# Convert numeric columns to float
for col in emb_cols:
    df[col] = df[col].astype(float)

# Create embeddings matrix for similarity ranking
embeddings_matrix = np.vstack([df.loc[:, emb_cols].iloc[i].to_numpy() for i in range(len(df))])
embedding_dim = embeddings_matrix.shape[1]
print(f"Embedding dimension: {embedding_dim}\n")

print(f"Final DataFrame shape: {df.shape}")
print(df.head())


COMPLETIONS_URL = "http://localhost:11434/v1/completions"
LLM_MODEL = "llama3.2"


def call_llm(prompt: str) -> str:
    try:
        response = requests.post(
            COMPLETIONS_URL,
            json={"model": LLM_MODEL, "prompt": prompt, "max_tokens": 512},
            timeout=60,
        )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "LLM request timed out. The prompt may still be too large or the model endpoint is slow. "
            "Try a shorter query or verify the server is responsive."
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"LLM request failed with status {response.status_code}: {response.text}"
        )
    data = response.json()
    if isinstance(data, dict):
        # Ollama v1 completions return text in choices[0]["text"]
        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            return data["choices"][0].get("text", "").strip()
        if "text" in data:
            return data["text"].strip()
    return str(data)


def find_similar_chunks(query: str, top_k: int = 5) -> pd.DataFrame:
    query_embedding = request_embedding(query)
    if query_embedding.shape[0] != embedding_dim:
        raise ValueError(
            f"Query embedding dimension {query_embedding.shape[0]} does not match chunk embedding dimension {embedding_dim}"
        )

    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings_matrix)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = df.iloc[top_indices].copy()
    results = results.head(MAX_CONTEXT_CHUNKS)
    results["similarity"] = similarities[top_indices][: len(results)]
    return results


if __name__ == "__main__":
    query = input("Enter a query to check cosine similarity: ").strip()
    if query:
        print(f"\nFinding top similar chunks for query: {query}\n")
        results = find_similar_chunks(query, top_k=5)
        print(results[["tutorial_number", "tutorial_name", "chunk_id", "similarity", "text"]])

        prompt = build_rag_prompt(query, results)
        print("\nSending prompt to llama3.2...\n")
        answer = call_llm(prompt)
        print("\nLLM answer:\n")
        print(answer)
    else:
        print("No query entered; skipping similarity search.")

print(f"Saving CSV: {CSV_FILE}")
df.to_csv(CSV_FILE, index=False)
print(f"Saved CSV: {CSV_FILE}")

# Optional: if you want parquet support, uncomment below and install pyarrow or fastparquet.
# try:
#     df.to_parquet(PARQUET_FILE)
#     print(f"Saved parquet: {PARQUET_FILE}")
# except (ImportError, ValueError) as e:
#     print(f"Parquet save failed: {e}")
