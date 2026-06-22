#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def chunk_words(words, chunk_size, overlap):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks


def parse_filename(filename: str) -> tuple[str, str, str]:
    base = Path(filename).stem
    tutorial_number, sep, tutorial_name = base.partition(" - ")
    if not sep:
        tutorial_number = ""
        tutorial_name = base
    return base, tutorial_number, tutorial_name


def load_transcript(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    return normalize_text(text)


def write_chunks(output_dir: Path, base_name: str, chunks: list[str]) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_name = f"{base_name}_chunk_{index:03d}.txt"
        chunk_path = output_dir / chunk_name
        chunk_path.write_text(chunk, encoding="utf-8")
        metadata.append({
            "chunk_file": chunk_name,
            "chunk_id": index,
            "chunk_text": chunk,
        })

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Split transcript text files into overlapping chunks.")
    parser.add_argument("--input", default="transcripts", help="Directory containing transcript .txt files")
    parser.add_argument("--output", default="transcript_chunks", help="Directory to save chunk files")
    parser.add_argument("--chunk-size", type=int, default=300, help="Maximum number of words per chunk")
    parser.add_argument("--overlap", type=int, default=50, help="Number of overlapping words between chunks")
    parser.add_argument("--json", default="chunks.json", help="JSON file to save chunk metadata")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    json_path = output_dir / args.json
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_files = sorted(input_dir.glob("*.txt"))
    if not transcript_files:
        raise SystemExit(f"No .txt transcript files found in {input_dir}")

    all_chunks = []
    for transcript_path in transcript_files:
        text = load_transcript(transcript_path)
        base_name, tutorial_number, tutorial_name = parse_filename(transcript_path.name)

        if not text:
            print(f"Skipping empty transcript: {transcript_path.name}")
            continue

        words = text.split()
        chunks = chunk_words(words, args.chunk_size, args.overlap)
        if not chunks:
            print(f"No chunks created for: {transcript_path.name}")
            continue

        transcript_output_dir = output_dir / base_name
        transcript_output_dir.mkdir(exist_ok=True)
        metadata = write_chunks(transcript_output_dir, base_name, chunks)

        for chunk_meta in metadata:
            all_chunks.append({
                "filename": transcript_path.name,
                "file_name": base_name,
                "tutorial_number": tutorial_number,
                "tutorial_name": tutorial_name,
                "chunk_id": chunk_meta["chunk_id"],
                "chunk_file": str(transcript_output_dir.name + "/" + chunk_meta["chunk_file"]),
                "text": chunk_meta["chunk_text"],
            })

        print(f"Created {len(chunks)} chunks for {transcript_path.name}")

    json_path.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metadata for {len(all_chunks)} chunks to {json_path}")


if __name__ == "__main__":
    main()
