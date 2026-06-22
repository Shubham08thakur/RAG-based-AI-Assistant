import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None

# Paths
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
VIDEOS_DIR = BASE_DIR / "videos"
AUDIOS_DIR = BASE_DIR / "audios"
CHUNKS_DIR = BASE_DIR / "transcript_chunks"
CHUNKS_JSON = CHUNKS_DIR / "chunks.json"
CSV_FILE = BASE_DIR / "chunks_with_embeddings.csv"
STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
TRANSCRIPTS_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)
AUDIOS_DIR.mkdir(exist_ok=True)
CHUNKS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="RAG Teaching Assistant API")

# Gemini Key Check
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Configuration State
default_config = {
    "embeddings_model": "bge-m3" if not GEMINI_API_KEY else "text-embedding-004",
    "ollama_url": "http://localhost:11434/api/embeddings",
    "completions_url": "http://localhost:11434/v1/completions",
    "llm_model": "llama3.2" if not GEMINI_API_KEY else "gemini-1.5-flash",
    "max_context_chunks": 1,
    "max_chunk_characters": 800,
    "chunk_size": 300,
    "overlap": 50
}

def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return {**default_config, **json.load(f)}
        except Exception:
            return default_config
    return default_config

def save_config(config: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

# Global variables for caching loaded data
chunks_cache: List[dict] = []
embeddings_matrix: Optional[np.ndarray] = None
embeddings_df: Optional[pd.DataFrame] = None

# Background process manager state
class ProcessManager:
    def __init__(self):
        self.status = "idle"  # idle, processing
        self.current_task = ""
        self.progress = 0.0  # 0.0 to 1.0
        self.logs = []
        self.lock = threading.Lock()

    def log(self, message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        with self.lock:
            self.logs.append(log_line)
            if len(self.logs) > 500:
                self.logs.pop(0)
        print(log_line)

    def set_status(self, status: str, task: str = "", progress: float = 0.0):
        with self.lock:
            self.status = status
            self.current_task = task
            self.progress = progress

    def get_state(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "current_task": self.current_task,
                "progress": round(self.progress * 100, 1),
                "logs": self.logs[-50:]
            }

    def clear_logs(self):
        with self.lock:
            self.logs = []

pm = ProcessManager()

def reload_chunks_and_embeddings():
    global chunks_cache, embeddings_matrix, embeddings_df
    pm.log("Reloading chunks and embeddings...")
    
    # Load chunks.json
    if CHUNKS_JSON.exists():
        try:
            with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
                chunks_cache = json.load(f)
            pm.log(f"Loaded {len(chunks_cache)} chunks from chunks.json")
        except Exception as e:
            pm.log(f"Error loading chunks.json: {e}")
            chunks_cache = []
    else:
        pm.log("chunks.json does not exist yet.")
        chunks_cache = []

    # Load CSV embeddings
    if CSV_FILE.exists():
        try:
            df = pd.read_csv(CSV_FILE)
            emb_cols = [col for col in df.columns if col.startswith("embedding_")]
            
            # Check dimension compatibility if Gemini is enabled (requires 768 dimensions)
            expected_dim = 768 if GEMINI_API_KEY else None
            
            if emb_cols:
                csv_dim = len(emb_cols)
                if expected_dim is not None and csv_dim != expected_dim:
                    pm.log(f"Dimension mismatch: CSV is {csv_dim}-dim, but Gemini requires {expected_dim}-dim. Re-computation needed.")
                    embeddings_df = None
                    embeddings_matrix = None
                elif len(df) == len(chunks_cache) and set(df.get("chunk_file", [])) == set(c.get("chunk_file") for c in chunks_cache):
                    embeddings_df = df
                    embeddings_matrix = np.vstack([df.loc[:, emb_cols].iloc[i].to_numpy() for i in range(len(df))])
                    pm.log(f"Loaded embedding matrix of shape {embeddings_matrix.shape} from CSV")
                else:
                    embeddings_df = None
                    embeddings_matrix = None
                    pm.log("Saved CSV embeddings found, but they do not match current chunks. Re-computation needed.")
            else:
                embeddings_df = None
                embeddings_matrix = None
        except Exception as e:
            pm.log(f"Error loading embeddings CSV: {e}")
            embeddings_df = None
            embeddings_matrix = None
    else:
        embeddings_df = None
        embeddings_matrix = None
        pm.log("No embeddings CSV file found.")

# Initial data load
reload_chunks_and_embeddings()


def request_embedding(text: str, config: dict) -> np.ndarray:
    if GEMINI_API_KEY:
        # Use Google Gemini API for Embeddings
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GEMINI_API_KEY}"
        response = requests.post(
            url,
            json={
                "model": "models/text-embedding-004",
                "content": {
                    "parts": [{"text": text}]
                }
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Gemini Embedding failed with status {response.status_code}: {response.text}"
            )
        data = response.json()
        embedding = data.get("embedding", {}).get("values", [])
        if not embedding:
            raise ValueError(f"Received empty embedding from Gemini: {data}")
        return np.array(embedding, dtype=float)
    else:
        # Use Local Ollama API
        response = requests.post(
            config["ollama_url"],
            json={"prompt": text, "model": config["embeddings_model"]},
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Embedding request failed with status {response.status_code}: {response.text}"
            )
        data = response.json()
        embedding = data.get("embeddings") or data.get("embedding") or []
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("Received empty or invalid embedding from Ollama")
        return np.array(embedding, dtype=float)


def compute_all_embeddings_background():
    global embeddings_matrix, embeddings_df
    pm.set_status("processing", "Generating Embeddings", 0.0)
    config = load_config()
    
    try:
        if not CHUNKS_JSON.exists():
            pm.log("Error: chunks.json does not exist. Cannot compute embeddings.")
            pm.set_status("idle")
            return

        with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            pm.log("Error: chunks.json is empty.")
            pm.set_status("idle")
            return

        df = pd.DataFrame(chunks)
        model_name = "Gemini text-embedding-004" if GEMINI_API_KEY else config['embeddings_model']
        pm.log(f"Starting embeddings generation for {len(chunks)} chunks using model {model_name}...")
        
        # Check if we have some existing partial embeddings we can load
        existing_df = None
        if CSV_FILE.exists():
            try:
                existing_df = pd.read_csv(CSV_FILE)
            except Exception:
                pass
        
        # Determine embedding dimension
        pm.log("Requesting sample embedding to determine dimensionality...")
        sample_emb = request_embedding("test", config)
        emb_dim = sample_emb.shape[0]
        emb_cols = [f"embedding_{i}" for i in range(emb_dim)]
        
        # Pre-fill DataFrame
        for col in emb_cols:
            df[col] = np.nan
            
        # Copy existing embeddings if they match dimensions and files
        copied_count = 0
        if existing_df is not None:
            existing_emb_cols = [col for col in existing_df.columns if col.startswith("embedding_")]
            if len(existing_emb_cols) == emb_dim:
                for idx, row in df.iterrows():
                    chunk_file = row["chunk_file"]
                    match = existing_df[existing_df["chunk_file"] == chunk_file]
                    if not match.empty:
                        for dim, col in enumerate(emb_cols):
                            df.at[idx, col] = match.iloc[0][existing_emb_cols[dim]]
                        copied_count += 1
                pm.log(f"Reused {copied_count} existing embeddings from CSV cache.")

        # Compute missing embeddings
        for idx, row in df.iterrows():
            if pd.isna(df.at[idx, emb_cols[0]]):
                text = row["text"]
                pm.log(f"Computing embedding for chunk {idx + 1}/{len(df)}...")
                embedding = request_embedding(text, config)
                for dim, col in enumerate(emb_cols):
                    df.at[idx, col] = embedding[dim]
                # Sleep a little to respect rate limits
                time.sleep(0.1 if GEMINI_API_KEY else 0.05)
                
            pm.set_status("processing", "Generating Embeddings", (idx + 1) / len(df))

        # Save to CSV
        df.to_csv(CSV_FILE, index=False)
        pm.log(f"Successfully saved embeddings for {len(df)} chunks to {CSV_FILE}")
        
        # Reload cache
        reload_chunks_and_embeddings()
        pm.log("Embeddings computation completed successfully!")
    except Exception as e:
        pm.log(f"Error during embedding generation: {e}")
    finally:
        pm.set_status("idle")


def run_pipeline_step_background(step_name: str, cmd_args: List[str]):
    pm.set_status("processing", step_name, 0.1)
    pm.log(f"Starting pipeline step: {step_name}")
    pm.log(f"Running command: {' '.join(cmd_args)}")
    
    try:
        process = subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            pm.log(line.strip())
            
        process.wait()
        
        if process.returncode == 0:
            pm.log(f"Step {step_name} completed successfully.")
            if "create_chunk.py" in cmd_args[1] or "create_chunk" in cmd_args[0]:
                reload_chunks_and_embeddings()
        else:
            pm.log(f"Step {step_name} failed with exit code {process.returncode}")
    except Exception as e:
        pm.log(f"Exception during step {step_name}: {e}")
    finally:
        pm.set_status("idle")


# Models for API
class QueryRequest(BaseModel):
    query: str

class ConfigUpdateRequest(BaseModel):
    embeddings_model: str
    ollama_url: str
    completions_url: str
    llm_model: str
    max_context_chunks: int
    max_chunk_characters: int
    chunk_size: int
    overlap: int

class ProcessRequest(BaseModel):
    action: str


# API Endpoints
@app.get("/api/status")
def get_status():
    num_transcripts = len(list(TRANSCRIPTS_DIR.glob("*.txt")))
    num_videos = len(list(VIDEOS_DIR.glob("*.mp4")))
    num_audios = len(list(AUDIOS_DIR.glob("*.mp3")))
    num_chunks = len(chunks_cache)
    
    embeddings_loaded = embeddings_matrix is not None
    emb_count = 0
    emb_dim = 0
    if embeddings_loaded:
        emb_count = embeddings_matrix.shape[0]
        emb_dim = embeddings_matrix.shape[1]

    expected_dim = 768 if GEMINI_API_KEY else 1024
    needs_embeddings = (num_chunks > 0 and 
                        (emb_count != num_chunks or emb_dim != expected_dim))

    return {
        "transcripts_count": num_transcripts,
        "videos_count": num_videos,
        "audios_count": num_audios,
        "chunks_count": num_chunks,
        "embeddings_loaded": embeddings_loaded,
        "embeddings_count": emb_count,
        "embeddings_dimension": emb_dim,
        "needs_embeddings": needs_embeddings,
        "is_gemini_enabled": GEMINI_API_KEY is not None,
        "job": pm.get_state()
    }


@app.get("/api/config")
def get_config():
    return load_config()


@app.post("/api/config")
def update_config(req: ConfigUpdateRequest):
    save_config(req.dict())
    return {"status": "success", "message": "Configuration updated successfully"}


@app.post("/api/query")
def run_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    config = load_config()
    
    if not chunks_cache:
        raise HTTPException(status_code=400, detail="No chunks loaded. Please ingest and chunk some transcripts first.")
        
    # Check if dimensions match the expected target
    expected_dim = 768 if GEMINI_API_KEY else None
    
    if embeddings_matrix is None or len(chunks_cache) != embeddings_matrix.shape[0]:
        raise HTTPException(status_code=400, detail="Embeddings are not generated or mismatch chunk count. Please generate embeddings first.")

    if expected_dim is not None and embeddings_matrix.shape[1] != expected_dim:
        raise HTTPException(status_code=400, detail=f"Embeddings dimension mismatch. Expected {expected_dim} for Gemini, but loaded CSV contains {embeddings_matrix.shape[1]}. Please regenerate embeddings.")

    if cosine_similarity is None:
        raise HTTPException(status_code=500, detail="scikit-learn is not installed in the backend environment.")

    try:
        # Get query embedding
        query_emb = request_embedding(req.query, config)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_emb.reshape(1, -1), embeddings_matrix)[0]
        
        # Get top matching indices
        top_k = min(5, len(chunks_cache))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Gather context
        retrieved_chunks = []
        max_context = config["max_context_chunks"]
        max_chars = config["max_chunk_characters"]
        
        for idx in top_indices:
            chunk = chunks_cache[idx]
            sim = float(similarities[idx])
            retrieved_chunks.append({
                "tutorial_number": chunk.get("tutorial_number", ""),
                "tutorial_name": chunk.get("tutorial_name", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "chunk_file": chunk.get("chunk_file", ""),
                "text": chunk.get("text", ""),
                "similarity": sim
            })
            
        context_chunks = retrieved_chunks[:max_context]
        
        # Build prompt
        chunk_entries = []
        for c in context_chunks:
            source = f"[{c['tutorial_number']}] {c['tutorial_name']} chunk {c['chunk_id']}"
            text = c['text'].strip()
            if len(text) > max_chars:
                text = text[:max_chars].rstrip() + "..."
            chunk_entries.append(f"{source}\n{text}")

        retrieved_text = "\n\n".join(chunk_entries)
        
        system_prompt = (
            "You are a retrieval-augmented AI assistant. Use only the retrieved transcript chunks "
            "to answer the user question. Do not invent information or hallucinate. "
            "If the answer is not present in the provided chunks, respond with: "
            "'I don't know based on the provided sources.'"
        )
        
        prompt = (
            f"{system_prompt}\n\n"
            "Retrieved chunks:\n"
            f"{retrieved_text}\n\n"
            f"User question: {req.query}\n\n"
            "Answer:"
        )
        
        # Call LLM (Gemini or Ollama)
        if GEMINI_API_KEY:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
            response = requests.post(
                url,
                json={
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }]
                },
                timeout=60,
            )
            if response.status_code != 200:
                raise RuntimeError(f"Gemini API completions request failed with status {response.status_code}: {response.text}")
                
            data = response.json()
            answer = ""
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    answer = parts[0].get("text", "").strip()
            if not answer:
                answer = "Error: Received empty response from Gemini API"
        else:
            response = requests.post(
                config["completions_url"],
                json={"model": config["llm_model"], "prompt": prompt, "max_tokens": 512},
                timeout=60,
            )
            if response.status_code != 200:
                raise RuntimeError(f"LLM endpoint returned status {response.status_code}: {response.text}")
                
            data = response.json()
            answer = ""
            if isinstance(data, dict):
                if "choices" in data and data["choices"]:
                    answer = data["choices"][0].get("text", "").strip()
                elif "text" in data:
                    answer = data["text"].strip()
            else:
                answer = str(data)

        return {
            "query": req.query,
            "answer": answer,
            "prompt": prompt,
            "retrieved_chunks": retrieved_chunks
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query execution failed: {str(e)}")


@app.get("/api/chunks")
def list_chunks(q: Optional[str] = None, page: int = 1, limit: int = 20):
    if not chunks_cache:
        return {"chunks": [], "total": 0, "page": page, "limit": limit}
        
    filtered = chunks_cache
    if q and q.strip():
        search_query = q.lower().strip()
        filtered = [
            c for c in chunks_cache 
            if search_query in c.get("text", "").lower() or 
               search_query in c.get("tutorial_name", "").lower() or 
               search_query in c.get("tutorial_number", "").lower()
        ]
        
    total = len(filtered)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    
    paginated = filtered[start_idx:end_idx]
    
    # Append similarity caching info if we have embeddings
    expected_dim = 768 if GEMINI_API_KEY else None
    
    for c in paginated:
        has_emb = False
        if embeddings_df is not None:
            match = embeddings_df[embeddings_df["chunk_file"] == c["chunk_file"]]
            if not match.empty:
                # verify dim
                emb_cols = [col for col in match.columns if col.startswith("embedding_")]
                if expected_dim is None or len(emb_cols) == expected_dim:
                    has_emb = True
        c["has_embedding"] = has_emb
        
    return {
        "chunks": paginated,
        "total": total,
        "page": page,
        "limit": limit
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Empty filename")
        
    if filename.lower().endswith(".txt"):
        save_path = TRANSCRIPTS_DIR / filename
        content = await file.read()
        save_path.write_bytes(content)
        pm.log(f"Uploaded transcript saved to: {save_path.name}")
        return {"status": "success", "message": f"Transcript {filename} uploaded successfully."}
        
    elif filename.lower().endswith(".mp4"):
        save_path = VIDEOS_DIR / filename
        content = await file.read()
        save_path.write_bytes(content)
        pm.log(f"Uploaded video saved to: {save_path.name}")
        return {"status": "success", "message": f"Video {filename} uploaded successfully."}
        
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload .txt or .mp4 files.")


@app.post("/api/process/run")
def trigger_process(req: ProcessRequest, background_tasks: BackgroundTasks):
    if pm.status == "processing":
        raise HTTPException(status_code=400, detail=f"A background process is already running: {pm.current_task}")
        
    config = load_config()
    python_bin = str(BASE_DIR / "venv" / "bin" / "python")
    if not os.path.exists(python_bin):
        python_bin = "python"
        
    action = req.action.lower()
    pm.clear_logs()

    if action == "convert":
        background_tasks.add_task(
            run_pipeline_step_background,
            "Extracting Audio from Videos (MP4 -> MP3)",
            [python_bin, str(BASE_DIR / "process_video.py")]
        )
        return {"status": "success", "message": "Audio extraction started."}
        
    elif action == "transcribe":
        background_tasks.add_task(
            run_pipeline_step_background,
            "Transcribing Audio to Text (Whisper)",
            [python_bin, str(BASE_DIR / "stt.py")]
        )
        return {"status": "success", "message": "Transcription process started."}
        
    elif action == "chunk":
        chunk_args = [
            python_bin,
            str(BASE_DIR / "create_chunk.py"),
            "--input", "transcripts",
            "--output", "transcript_chunks",
            "--chunk-size", str(config["chunk_size"]),
            "--overlap", str(config["overlap"]),
            "--json", "chunks.json"
        ]
        background_tasks.add_task(
            run_pipeline_step_background,
            "Splitting Transcripts into Chunks",
            chunk_args
        )
        return {"status": "success", "message": "Transcript chunking started."}
        
    elif action == "embeddings":
        background_tasks.add_task(compute_all_embeddings_background)
        return {"status": "success", "message": "Embeddings computation started."}
        
    elif action == "all":
        def run_full_pipeline():
            pm.log("Starting full ingestion pipeline...")
            run_pipeline_step_background("1. Extracting Audio (MP4 -> MP3)", [python_bin, str(BASE_DIR / "process_video.py")])
            if pm.status == "processing":
                return
            run_pipeline_step_background("2. Transcribing Audio (Whisper)", [python_bin, str(BASE_DIR / "stt.py")])
            if pm.status == "processing":
                return
            chunk_args = [
                python_bin,
                str(BASE_DIR / "create_chunk.py"),
                "--input", "transcripts",
                "--output", "transcript_chunks",
                "--chunk-size", str(config["chunk_size"]),
                "--overlap", str(config["overlap"]),
                "--json", "chunks.json"
            ]
            run_pipeline_step_background("3. Splitting Transcripts", chunk_args)
            if pm.status == "processing":
                return
            compute_all_embeddings_background()
            
        background_tasks.add_task(run_full_pipeline)
        return {"status": "success", "message": "Full ingestion pipeline started."}
        
    else:
        raise HTTPException(status_code=400, detail=f"Unknown process action: {action}")


# Mount Static Files
static_path = BASE_DIR / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")
else:
    @app.get("/")
    def index():
        return HTMLResponse("<h1>RAG UI Backend Running</h1><p>Please create the static/ directory and index.html file.</p>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
