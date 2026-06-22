#!/bin/bash
# RAG Teaching Assistant Web UI Start Script

# Get directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Check if venv exists and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "WARNING: venv directory not found. Using system python."
fi

# Run FastAPI app via Uvicorn
echo "Starting RAG Web UI on http://localhost:8000 ..."
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
