from functools import lru_cache

import torch
from transformers import MarianMTModel, MarianTokenizer


MODEL_NAMES = {
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
}

SUPPORTED_LANGUAGES = {"en", "ar", "de"}


def normalize_language(language: str) -> str:
    language = language.strip().lower()

    if language not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported languages: {supported}"
        )

    return language


@lru_cache(maxsize=4)
def load_model(source_language: str, target_language: str):
    model_name = MODEL_NAMES.get(
        (source_language, target_language)
    )

    if model_name is None:
        raise ValueError(
            f"Translation direction is not supported: "
            f"{source_language} -> {target_language}"
        )

    print(f"Loading translation model: {model_name}")

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    print(
        f"Translation model ready: "
        f"{source_language} -> {target_language} on {device}"
    )

    return tokenizer, model, device


def translate_direct(
    text: str,
    source_language: str,
    target_language: str,
) -> str:
    tokenizer, model, device = load_model(
        source_language,
        target_language,
    )

    encoded = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=256,
            num_beams=4,
        )

    return tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0].strip()


def translate_text(
    text: str,
    source_language: str,
    target_language: str,
) -> str:
    cleaned_text = text.strip()

    if not cleaned_text:
        return ""

    source_language = normalize_language(source_language)
    target_language = normalize_language(target_language)

    if source_language == target_language:
        return cleaned_text

    direct_pair = (source_language, target_language)

    if direct_pair in MODEL_NAMES:
        return translate_direct(
            cleaned_text,
            source_language,
            target_language,
        )

    # Arabic <-> German goes through English.
    english_text = translate_direct(
        cleaned_text,
        source_language,
        "en",
    )

    return translate_direct(
        english_text,
        "en",
        target_language,
    )


if __name__ == "__main__":
    examples = [
        ("I love my family", "en", "ar"),
        ("I love my family", "en", "de"),
        ("أنا أحب عائلتي", "ar", "en"),
        ("Ich brauche Hilfe", "de", "en"),
        ("أنا أحتاج المساعدة", "ar", "de"),
    ]

    for text, source, target in examples:
        result = translate_text(
            text,
            source,
            target,
        )

        print()
        print(f"{source} -> {target}")
        print(f"Input: {text}")
        print(f"Output: {result}")