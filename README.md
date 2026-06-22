# RAG Teaching Assistant

This repository implements a simple Retrieval-Augmented Generation (RAG) teaching assistant using your own transcript data. It converts transcripts into overlapping chunks, computes embeddings, and uses a local LLM to answer user questions from retrieved source content.

## How it works

1. Convert your source data into transcript text files.
2. Split transcripts into overlapping chunks using `create_chunk.py`.
3. Generate embeddings and ask questions with `read_chunk.py`.
4. The assistant retrieves the most relevant transcript chunks and answers using only that source.

## Requirements

- Python 3.10+ (recommended)
- `pandas`
- `numpy`
- `requests`
- `scikit-learn`
- A local Ollama server or equivalent API for embeddings and completions

## Recommended environment setup

```bash
python -m venv venv
source venv/bin/activate
pip install pandas numpy requests scikit-learn
```

If you also want to transcribe audio or video files locally, install Whisper and dependencies:

```bash
pip install git+https://github.com/openai/whisper.git
pip install torch
```

## Prepare your own data

### Option A: Use existing transcript text files

Place your `.txt` transcript files in the `transcripts/` directory. Each file should contain the text for one source document or video.

Example file names:

- `transcripts/01 - Why Associations Choose MapleLMS ｜ Independent Review by Talented Learning.txt`
- `transcripts/02 - SHRM 26 - The Next Chapter of Workforce Transformation Begins..txt`

### Option B: Convert video files to audio and transcript them

1. Put your `.mp4` videos in the `videos/` folder.
2. Run `process_video.py` to convert them to `audios/*.mp3`.
3. Run `stt.py` to generate transcripts into `transcripts/*.txt`.

```bash
python process_video.py
python stt.py
```

## Create chunks from transcripts

Run the chunking script to split each transcript into smaller overlapping chunks.

```bash
python create_chunk.py --input transcripts --output transcript_chunks --chunk-size 300 --overlap 50 --json chunks.json
```

This creates:

- `transcript_chunks/<file_base>/<file_base>_chunk_001.txt`
- `transcript_chunks/<file_base>/<file_base>_chunk_002.txt`
- `transcript_chunks/chunks.json`

## Run the RAG assistant

The `read_chunk.py` script loads `transcript_chunks/chunks.json`, computes or reuses embeddings, finds the most relevant chunk for your query, and sends a prompt to the local LLM.

```bash
python read_chunk.py
```

Then enter your question when prompted.

### What `read_chunk.py` does

- Loads chunk metadata from `transcript_chunks/chunks.json`
- Requests embeddings from a local Ollama embeddings API at `http://localhost:11434/api/embeddings`
- Computes cosine similarity between the query and transcript chunks
- Retrieves the most relevant chunk(s)
- Builds a RAG prompt and sends it to a local LLM at `http://localhost:11434/v1/completions`
- Prints the final answer

## Local Ollama setup

The current scripts assume Ollama is running locally on port `11434` and provides:

- Embeddings model: `bge-m3`
- LLM model: `llama3.2`

If you use a different model or service, update `read_chunk.py` accordingly:

- `EMBEDDINGS_MODEL`
- `OLLAMA_URL`
- `COMPLETIONS_URL`
- `LLM_MODEL`

## Tips for better results

- Keep your transcript text clean and well-formatted.
- Use an appropriate chunk size and overlap for your content.
- Only ask questions that can be answered from your transcripts.
- If the answer is not found in the retrieved context, the assistant should say it does not know.

## Directory overview

- `transcripts/` - source transcript text files
- `transcript_chunks/` - generated chunk text files and `chunks.json`
- `create_chunk.py` - creates overlapping transcript chunks
- `read_chunk.py` - runs RAG retrieval and LLM answer generation
- `process_video.py` - converts `videos/` `.mp4` files to `audios/` `.mp3`
- `stt.py` - transcribes audio files to `transcripts/`

## Troubleshooting

- If no `.txt` files are found, verify your `transcripts/` directory contains valid text files.
- If embeddings fail, verify the Ollama server is running and reachable at `http://localhost:11434`.
- If the completion endpoint fails, check the model names and API URL.

## Customization

To use a different dataset, simply replace the files in `transcripts/` and rerun the chunk creation step. The assistant will then use your own data for retrieval and answers.
