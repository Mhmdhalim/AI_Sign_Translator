from flask import Flask, jsonify, request

from translator import translate_text

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "translation",
        }
    )


@app.post("/translate")
def translate():
    data = request.get_json(silent=True) or {}

    text = str(data.get("text", "")).strip()
    source_language = str(
        data.get("source_language", "en")
    ).strip()
    target_language = str(
        data.get("target_language", "ar")
    ).strip()

    if not text:
        return (
            jsonify(
                {
                    "error": "Text is required.",
                }
            ),
            400,
        )

    try:
        translated_text = translate_text(
            text=text,
            source_language=source_language,
            target_language=target_language,
        )

        return jsonify(
            {
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
                "translation": translated_text,
            }
        )

    except ValueError as error:
        return (
            jsonify(
                {
                    "error": str(error),
                }
            ),
            400,
        )

    except Exception as error:
        print(f"Translation error: {error}")

        return (
            jsonify(
                {
                    "error": "Translation failed.",
                }
            ),
            500,
        )

if __name__ == "__main__":
    print("Translation service is ready.")
    print("Models will load automatically on the first request.")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=False,
    )