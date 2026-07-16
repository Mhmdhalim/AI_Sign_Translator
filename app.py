import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

RUNTIME_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
TRAINING_PYTHON = PROJECT_ROOT / "train_env" / "Scripts" / "python.exe"
TRANSLATION_PYTHON = PROJECT_ROOT / "translation_env" / "Scripts" / "python.exe"
SPEECH_PYTHON = PROJECT_ROOT / "speech_env" / "Scripts" / "python.exe"

LETTER_RECOGNIZER = (
    PROJECT_ROOT
    / "realtime"
    / "realtime_predict.py"
)

WORD_RECOGNIZER = (
    PROJECT_ROOT
    / "realtime"
    / "realtime_word_predict.py"
)

TRANSLATION_SERVICE = (
    PROJECT_ROOT
    / "translation"
    / "service.py"
)

SPEECH_RECOGNIZER = (
    PROJECT_ROOT
    / "speech_to_sign"
    / "speech_recognizer.py"
)


translation_process = None


def check_file(path: Path, description: str) -> bool:
    if path.exists():
        return True

    print(f"\nMissing {description}:")
    print(path)
    return False


def run_script(
    python_path: Path,
    script_path: Path,
    description: str,
) -> None:
    if not check_file(python_path, f"{description} Python environment"):
        return

    if not check_file(script_path, description):
        return

    print()
    print(f"Starting {description}...")
    print("Close its window or press Q to return.")

    try:
        subprocess.run(
            [
                str(python_path),
                str(script_path),
            ],
            cwd=str(PROJECT_ROOT),
            check=False,
        )

    except KeyboardInterrupt:
        print(f"\n{description} stopped.")

    except OSError as error:
        print(f"\nCould not start {description}: {error}")


def is_translation_service_running() -> bool:
    global translation_process

    return (
        translation_process is not None
        and translation_process.poll() is None
    )


def start_translation_service() -> bool:
    global translation_process

    if is_translation_service_running():
        print("Translation service is already running.")
        return True

    if not check_file(
        TRANSLATION_PYTHON,
        "translation environment",
    ):
        return False

    if not check_file(
        TRANSLATION_SERVICE,
        "translation service",
    ):
        return False

    print()
    print("Starting translation service...")
    print("The first translation request may take some time.")

    try:
        translation_process = subprocess.Popen(
            [
                str(TRANSLATION_PYTHON),
                str(TRANSLATION_SERVICE),
            ],
            cwd=str(PROJECT_ROOT),
        )

        time.sleep(3)

        if translation_process.poll() is not None:
            print("Translation service stopped unexpectedly.")
            translation_process = None
            return False

        print("Translation service started.")
        return True

    except OSError as error:
        print(f"Could not start translation service: {error}")
        translation_process = None
        return False


def stop_translation_service() -> None:
    global translation_process

    if not is_translation_service_running():
        translation_process = None
        return

    print("Stopping translation service...")

    translation_process.terminate()

    try:
        translation_process.wait(timeout=5)

    except subprocess.TimeoutExpired:
        translation_process.kill()
        translation_process.wait(timeout=5)

    translation_process = None
    print("Translation service stopped.")


def run_letter_recognition() -> None:
    run_script(
        python_path=RUNTIME_PYTHON,
        script_path=LETTER_RECOGNIZER,
        description="ASL letter recognition",
    )


def run_word_recognition() -> None:
    if not start_translation_service():
        print(
            "Word recognition will not start because "
            "the translation service is unavailable."
        )
        return

    run_script(
        python_path=RUNTIME_PYTHON,
        script_path=WORD_RECOGNIZER,
        description="ASL word recognition and translation",
    )


def run_speech_recognition() -> None:
    run_script(
        python_path=SPEECH_PYTHON,
        script_path=SPEECH_RECOGNIZER,
        description="multilingual speech recognition",
    )


def show_system_status() -> None:
    items = [
        (
            "Runtime environment",
            RUNTIME_PYTHON,
        ),
        (
            "Training environment",
            TRAINING_PYTHON,
        ),
        (
            "Translation environment",
            TRANSLATION_PYTHON,
        ),
        (
            "Speech environment",
            SPEECH_PYTHON,
        ),
        (
            "Letter recognizer",
            LETTER_RECOGNIZER,
        ),
        (
            "Word recognizer",
            WORD_RECOGNIZER,
        ),
        (
            "Translation service",
            TRANSLATION_SERVICE,
        ),
        (
            "Speech recognizer",
            SPEECH_RECOGNIZER,
        ),
    ]

    print()
    print("=" * 55)
    print("SYSTEM STATUS")
    print("=" * 55)

    for name, path in items:
        status = "OK" if path.exists() else "MISSING"
        print(f"{name:<30} {status}")

    service_status = (
        "RUNNING"
        if is_translation_service_running()
        else "STOPPED"
    )

    print(f"{'Translation server':<30} {service_status}")
    print("=" * 55)


def print_menu() -> None:
    print()
    print("=" * 55)
    print("AI SIGN TRANSLATOR")
    print("=" * 55)
    print("1. Sign letters -> English speech")
    print("2. Sign words -> English / Arabic / German")
    print("3. Speech -> Text")
    print("4. Start translation service")
    print("5. Stop translation service")
    print("6. Show system status")
    print("0. Exit")
    print("=" * 55)


def main() -> None:
    print("AI Sign Translator started.")

    try:
        while True:
            print_menu()

            choice = input("Choose an option: ").strip()

            if choice == "1":
                run_letter_recognition()

            elif choice == "2":
                run_word_recognition()

            elif choice == "3":
                run_speech_recognition()

            elif choice == "4":
                start_translation_service()

            elif choice == "5":
                stop_translation_service()

            elif choice == "6":
                show_system_status()

            elif choice == "0":
                print("Closing AI Sign Translator...")
                break

            else:
                print("Invalid option. Choose a number from the menu.")

    except KeyboardInterrupt:
        print("\nApplication interrupted.")

    finally:
        stop_translation_service()
        print("Application closed.")


if __name__ == "__main__":
    main()