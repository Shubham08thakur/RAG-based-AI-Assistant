import whisper
import os
import time
import json

AUDIO_DIR = "audios"
OUTPUT_DIR = "transcripts"
OUTPUT_JSON = "output.json"
MODEL_NAME = "large-v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading {MODEL_NAME} model (CPU)... this can take a minute.")
model = whisper.load_model(MODEL_NAME)
print("Model loaded.\n")

mp3_files = sorted(f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".mp3"))
all_results = []

for filename in mp3_files:
    audio_path = os.path.join(AUDIO_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + ".txt")
    file_name, _ = os.path.splitext(filename)
    tutorial_number, sep, tutorial_name = file_name.partition(" - ")
    if not sep:
        tutorial_number = ""
        tutorial_name = file_name

    if os.path.exists(output_path):
        print(f"Skipping (already done): {filename}")
        with open(output_path, "r", encoding="utf-8") as f:
            transcript = f.read()
        all_results.append({
            "filename": filename,
            "file_name": file_name,
            "tutorial_number": tutorial_number,
            "tutorial_name": tutorial_name,
            "transcript": transcript
        })
        continue

    print(f"Transcribing: {filename}")
    start = time.time()

    try:
        result = model.transcribe(audio_path, fp16=False)
        transcript = result["text"].strip()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        all_results.append({
            "filename": filename,
            "file_name": file_name,
            "tutorial_number": tutorial_number,
            "tutorial_name": tutorial_name,
            "transcript": transcript
        })
        print(f"  Done in {(time.time() - start) / 60:.1f} min -> {output_path}\n")
    except Exception as e:
        print(f"  FAILED: {filename} -> {e}\n")

# Save all results to JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"All results saved to: {OUTPUT_JSON}")
print("All files processed.")