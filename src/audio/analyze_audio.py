import librosa
from pathlib import Path
import subprocess
import tempfile
import os


def convert_to_wav(file_path):
    """Convert any unsupported audio format to WAV using FFmpeg."""

    file_path = Path(file_path)

    temp_wav = Path(tempfile.gettempdir()) / f"{file_path.stem}_voicesense.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(file_path),
        "-acodec",
        "pcm_s16le",
        str(temp_wav)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        error_message = result.stderr.decode(errors="ignore")
        raise RuntimeError(f"FFmpeg conversion failed:\n{error_message}")

    return temp_wav


def analyze_audio(file_path):

    converted_file = None

    try:
        file_path = Path(file_path)

        # Check file exists
        if not file_path.exists():
            print(f"Error: Audio file not found: {file_path}")
            return

        print(f"\nAnalyzing: {file_path.name}")

        # WAV can be loaded directly.
        # Other formats are converted using FFmpeg.
        if file_path.suffix.lower() == ".wav":
            audio_file = file_path
        else:
            print("Converting audio to WAV...")
            converted_file = convert_to_wav(file_path)
            audio_file = converted_file

        # Load audio
        audio, sample_rate = librosa.load(
            str(audio_file),
            sr=None,
            mono=False
        )

        # Determine channels and samples
        if audio.ndim == 1:
            channels = 1
            total_samples = len(audio)
        else:
            channels = audio.shape[0]
            total_samples = audio.shape[1]

        # Calculate duration
        duration = total_samples / sample_rate

        print("\n========== AUDIO INFORMATION ==========")
        print(f"File Name    : {file_path.name}")
        print(f"Duration     : {duration:.2f} seconds")
        print(f"Sample Rate  : {sample_rate} Hz")
        print(f"Channels     : {channels}")
        print(f"Samples      : {total_samples}")
        print("=======================================\n")

    except Exception as error:
        print("\nError while analyzing audio:")
        print(type(error).__name__)
        print(error)

    finally:
        # Delete temporary WAV
        if converted_file and converted_file.exists():
            try:
                os.remove(converted_file)
            except Exception:
                pass


if __name__ == "__main__":
    analyze_audio("data/audio/test.m4a")