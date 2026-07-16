from pathlib import Path
import tempfile
import wave

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 6

MODEL_SIZE = "small"

SUPPORTED_LANGUAGES = {
    "auto": None,
    "ar": "ar",
    "en": "en",
    "de": "de",
}


def select_language() -> tuple[str, str | None]:
    print()
    print("Choose spoken language:")
    print("0: Auto detect")
    print("1: Arabic")
    print("2: English")
    print("3: German")

    choices = {
        "0": ("auto", None),
        "1": ("ar", "ar"),
        "2": ("en", "en"),
        "3": ("de", "de"),
    }

    selected = input("Language number: ").strip()

    return choices.get(selected, ("auto", None))


def record_audio(duration: int = RECORD_SECONDS) -> np.ndarray:
    print()
    print(f"Recording for {duration} seconds...")
    print("Speak now.")

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )

    sd.wait()

    audio = np.asarray(audio, dtype=np.int16)

    print("Recording finished.")

    return audio


def save_wav(audio: np.ndarray, output_path: Path) -> None:
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio.tobytes())


def create_model() -> WhisperModel:
    print()
    print(f"Loading Whisper model: {MODEL_SIZE}")

    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8",
    )

    print("Speech recognition model is ready.")

    return model


def transcribe_audio(
    model: WhisperModel,
    audio_path: Path,
    language: str | None,
) -> tuple[str, str, float]:
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    text_parts = []

    for segment in segments:
        segment_text = segment.text.strip()

        if segment_text:
            text_parts.append(segment_text)

    recognized_text = " ".join(text_parts).strip()

    detected_language = str(info.language)
    language_probability = float(info.language_probability)

    return (
        recognized_text,
        detected_language,
        language_probability,
    )


def recognize_once(
    model: WhisperModel,
    language: str | None,
) -> tuple[str, str]:
    audio = record_audio()

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ) as temporary_file:
        audio_path = Path(temporary_file.name)

    try:
        save_wav(audio, audio_path)

        text, detected_language, probability = transcribe_audio(
            model,
            audio_path,
            language,
        )

        print()
        print(f"Detected language: {detected_language}")
        print(f"Language confidence: {probability * 100:.1f}%")
        print(f"Recognized text: {text or '[No speech detected]'}")

        return text, detected_language

    finally:
        audio_path.unlink(missing_ok=True)


def main() -> None:
    _, selected_language = select_language()
    model = create_model()

    print()
    print("Controls:")
    print("ENTER: record speech")
    print("Q: quit")

    while True:
        command = input("\nPress ENTER to record, or Q to quit: ").strip()

        if command.lower() == "q":
            break

        try:
            recognize_once(
                model=model,
                language=selected_language,
            )

        except sd.PortAudioError as error:
            print(f"Microphone error: {error}")
            print("Check Windows microphone permissions and input device.")

        except Exception as error:
            print(f"Speech recognition error: {error}")


if __name__ == "__main__":
    main()