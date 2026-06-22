import os
import subprocess

video_dir = "videos"
output_dir = "audios"
os.makedirs(output_dir, exist_ok=True)

files = sorted(os.listdir(video_dir))

for file in files:
    if not file.lower().endswith(".mp4"):
        continue

    file_name, _ = os.path.splitext(file)
    tutorial_number, sep, tutorial_name = file_name.partition(" - ")
    if not sep:
        tutorial_number = ""
        tutorial_name = file_name

    print(f"original_file: {file}")
    print(f"file_name: {file_name}")
    print(f"tutorial_number: {tutorial_number}")
    print(f"tutorial_name: {tutorial_name}")

    mp3_path = os.path.join(output_dir, f"{file_name}.mp3")
    subprocess.run([
        "ffmpeg",
        "-y",
        "-i",
        os.path.join(video_dir, file),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-q:a",
        "2",
        mp3_path,
    ], check=True)
    print(f"converted to: {mp3_path}\n")