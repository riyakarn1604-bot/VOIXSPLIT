from pyannote.audio import Pipeline


def diarize_audio(file_path):

    print("\nLoading speaker diarization model...")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1"
    )

    print("Analyzing:", file_path)
    print("Please wait...\n")

    output = pipeline(file_path)

    speakers = set()

    print("========== SPEAKER DIARIZATION ==========")

    for turn, speaker in output.speaker_diarization:
        speakers.add(speaker)

        print(
            f"{turn.start:.2f}s - "
            f"{turn.end:.2f}s : "
            f"{speaker}"
        )

    print("==========================================")

    print(f"\nNumber of speakers detected: {len(speakers)}")


if __name__ == "__main__":
    diarize_audio("data/audio/test.m4a")