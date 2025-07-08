import gc
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

# === Config ===
TARGET_LANGS = [
    "bg", "hr", "hu", "he", "lt", "ro", "sk", "sl", "tr", "cs"
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SHORT_STRING_WORDS = 3


# === Context handling ===
def contextualize(text: str) -> str:
    if len(text.split()) <= 1:
        return f"Label: {text}"
    return f"This label is: '{text}'"


def extract_decontextualized(translated: str) -> str:
    parts = translated.split(":")
    if len(parts) > 1:
        return parts[-1].strip().strip("'\"")
    return translated.strip()


def collect_short_strings(obj, short_strings):
    if isinstance(obj, dict):
        for v in obj.values():
            collect_short_strings(v, short_strings)
    elif isinstance(obj, list):
        for item in obj:
            collect_short_strings(item, short_strings)
    elif isinstance(obj, str) and len(obj.split()) <= MAX_SHORT_STRING_WORDS:
        short_strings.add(obj)


def batch_translate(strings, tokenizer, model):
    strings = sorted(strings)
    if not strings:
        return {}

    results = {}
    contextualized = [contextualize(s) for s in strings]
    BATCH_START = 8
    BATCH_MAX = 64

    i = 0
    while i < len(contextualized):
        batch_size = BATCH_START
        success = False
        while not success and batch_size >= 1:
            chunk = contextualized[i:i + batch_size]
            originals = strings[i:i + batch_size]

            try:
                tokenized = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                with torch.no_grad():
                    outputs = model.generate(**tokenized)

                for orig, raw in zip(originals, tokenizer.batch_decode(outputs, skip_special_tokens=True)):
                    results[orig] = extract_decontextualized(raw)

                del tokenized, outputs
                torch.cuda.empty_cache()
                gc.collect()

                i += batch_size
                if batch_size < BATCH_MAX:
                    batch_size += 4
                success = True

            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "allocation" in str(e):
                    batch_size //= 2
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    for orig in originals:
                        results[orig] = orig
                    success = True
                    i += batch_size

    return results


def translate_json(obj, short_map, tokenizer, model):
    if isinstance(obj, dict):
        return {k: translate_json(v, short_map, tokenizer, model) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [translate_json(i, short_map, tokenizer, model) for i in obj]
    elif isinstance(obj, str):
        if obj in short_map:
            return short_map[obj]
        context = contextualize(obj)
        batch = tokenizer([context], return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**batch)
        raw = tokenizer.decode(gen[0], skip_special_tokens=True)
        return extract_decontextualized(raw)
    return obj


def translate_file(file: Path, lang_code: str, tokenizer, model, output_dir: Path):
    output_path = output_dir / f"{lang_code}-{lang_code.upper()}" / file.name
    if output_path.exists():
        return f"⏩ Skipped {file.name} → {lang_code} (exists)"

    with file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    short_strings = set()
    collect_short_strings(data, short_strings)

    short_map = batch_translate(short_strings, tokenizer, model)
    translated = translate_json(data, short_map, tokenizer, model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        json.dump(translated, out, ensure_ascii=False, indent=2)

    del data, short_strings, short_map, translated
    torch.cuda.empty_cache()
    gc.collect()

    return f"✅ {file.name} → {lang_code}"


MODEL_OVERRIDES = {
    "bg": "Helsinki-NLP/opus-mt-tc-big-en-bg",
    "cs": "Helsinki-NLP/opus-mt-en-cs",
    "hu": "Helsinki-NLP/opus-mt-tc-big-en-hu",
    "ro": "Helsinki-NLP/opus-mt-en-ro",
    "sk": "Helsinki-NLP/opus-mt-en-sk",
    "tr": "Helsinki-NLP/opus-mt-tc-big-en-tr",
    "hr": "Helsinki-NLP/opus-mt-en-mul",  # ✅ fallback to multilingual
    "he": "Helsinki-NLP/opus-mt-en-he",
    "lt": "Helsinki-NLP/opus-mt-tc-big-en-lt",
    "sl": "Helsinki-NLP/opus-mt-en-mul"  # ✅ also multilingual
}


def translate_language(lang_code: str, input_files: list[Path], output_dir: Path):
    print(f"\n🔠 Loading model for {lang_code.upper()}")
    model_name = MODEL_OVERRIDES.get(lang_code, f"Helsinki-NLP/opus-mt-en-{lang_code}")
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
    except Exception as e:
        print(f"❌ Failed to load model {model_name}: {e}")
        if model_name != f"Helsinki-NLP/opus-mt-en-{lang_code}":
            print("❌ No further fallback available.")
            return
        fallback_model = MODEL_OVERRIDES.get(lang_code)
        if fallback_model:
            print(f"🔁 Trying fallback model: {fallback_model}")
            try:
                tokenizer = MarianTokenizer.from_pretrained(fallback_model)
                model = MarianMTModel.from_pretrained(fallback_model).to(DEVICE)
            except Exception as fallback_e:
                print(f"❌ Fallback model also failed: {fallback_e}")
                return
        else:
            print("❌ No fallback model defined.")
            return

    for file in tqdm(input_files, desc=f"[{lang_code}]"):
        result = translate_file(file, lang_code, tokenizer, model, output_dir)
        tqdm.write(result)

    del tokenizer, model
    torch.cuda.empty_cache()
    gc.collect()


def translate_all(input_dir: Path, output_dir: Path):
    input_files = list(input_dir.glob("*.json"))
    for lang_code in TARGET_LANGS:
        translate_language(lang_code, input_files, output_dir)


if __name__ == "__main__":
    input_dir = Path("en")
    output_dir = Path("translated")
    translate_all(input_dir, output_dir)
