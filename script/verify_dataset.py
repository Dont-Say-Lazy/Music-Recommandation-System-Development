from pathlib import Path
import librosa
import pandas as pd

PROJECT_ROOT = Path(r"C:\HKBU_FYP")

AUDIO_ROOT = PROJECT_ROOT / "dataset" / "fma_medium"
METADATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "fma_metadata"
    / "tracks.csv"
)

print("Audio folder:", AUDIO_ROOT)
print("Metadata file:", METADATA_PATH)

print("Audio folder exists:", AUDIO_ROOT.exists())
print("Metadata file exists:", METADATA_PATH.exists())

if not AUDIO_ROOT.exists():
    raise SystemExit(
        "Audio folder not found. Check dataset/fma_medium."
    )

if not METADATA_PATH.exists():
    raise SystemExit(
        "Metadata file not found. Check dataset/fma_metadata/tracks.csv."
    )

audio_files = list(AUDIO_ROOT.rglob("*.mp3"))

print("MP3 files found:", len(audio_files))
print("First five audio files:")

for audio_file in audio_files[:5]:
    print(" -", audio_file)

if len(audio_files) == 0:
    raise SystemExit(
        "No MP3 files found inside fma_medium."
    )

tracks = pd.read_csv(
    METADATA_PATH,
    index_col=0,
    header=[0, 1]
)

print("Metadata rows:", len(tracks))
print("Metadata columns:", tracks.shape[1])

test_audio = audio_files[0]

print("\nTesting audio file:")
print(test_audio)

try:
    y, sr = librosa.load(
        test_audio,
        sr=22050,
        mono=True,
        duration=30
    )

    print("Sample rate:", sr)
    print("Number of samples:", len(y))
    print(
        "Duration in seconds:",
        round(len(y) / sr, 2)
    )
    print("\nDataset verification passed.")

except Exception as error:
    print("\nUnable to read audio file.")
    print("Error:", error)
