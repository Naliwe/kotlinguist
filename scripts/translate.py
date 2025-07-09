import gc
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm

TARGET_LANGS = ["sl", "bg", "hr", "hu", "he", "lt", "ro", "sk", "tr", "cs"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SHORT_STRING_WORDS = 3
BATCH_START = 4
BATCH_MAX = 64


def contextualize(text: str, lang_code: str) -> str:
    model_cfg = MODEL_CONFIGS.get(lang_code)
    if model_cfg is None or model_cfg.prefix is None:
        return text.strip()
    return f">>{model_cfg.prefix}<< {text.strip()}"


def extract_decontextualized(translated: str, original: str) -> str:
    return original if looks_suspicious(translated, original) else translated.strip()


def looks_suspicious(translated: str, original: str) -> bool:
    if translated.lower() == original.lower():
        return True
    return False


def is_placeholder(text: str) -> bool:
    return bool(re.match(r"^[A-Z]{2}[A-Z0-9\sX]{5,}$", text.strip()))


def collect_short_strings(obj, short_strings):
    if isinstance(obj, dict):
        for v in obj.values():
            collect_short_strings(v, short_strings)
    elif isinstance(obj, list):
        for item in obj:
            collect_short_strings(item, short_strings)
    elif isinstance(obj, str) and len(obj.split()) <= MAX_SHORT_STRING_WORDS:
        short_strings.add(obj.strip())


def batch_translate(strings, tokenizer, model, lang_code):
    model_cfg = MODEL_CONFIGS.get(lang_code, ModelConfig("", batch=BATCH_MAX))
    adaptive_max = model_cfg.batch
    strings = sorted(strings)
    results = {}
    contextualized = [contextualize(s, lang_code) for s in strings]
    i = 0
    while i < len(strings):
        batch_size = min(BATCH_START, adaptive_max)
        success = False
        while not success and batch_size >= 1:
            batch_raw = strings[i:i + batch_size]
            batch_ctx = contextualized[i:i + batch_size]
            try:
                tokenized = tokenizer(batch_ctx, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                with torch.no_grad():
                    generate_args = {**tokenized, **model_cfg.generation_args(tokenizer)}
                outputs = model.generate(**generate_args)
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for orig, raw in zip(batch_raw, decoded):
                    results[orig] = extract_decontextualized(raw, orig)
                i += batch_size
                if batch_size < adaptive_max:
                    batch_size += 4
                success = True
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    batch_size //= 2
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    for orig in batch_raw:
                        results[orig] = orig
                    i += batch_size
                    success = True
    return results


def clean_string(text: str) -> str:
    text = re.sub(r'^"|"$', '', text)
    text = re.sub(r"^'|'$", '', text)
    text = re.sub(r'[-]{5,}', '-', text)
    text = re.sub(r'[.]{5,}', '.', text)
    return text.strip()


def collect_long_strings(obj, short_map, collected):
    if isinstance(obj, dict):
        for v in obj.values():
            collect_long_strings(v, short_map, collected)
    elif isinstance(obj, list):
        for i in obj:
            collect_long_strings(i, short_map, collected)
    elif isinstance(obj, str):
        stripped = obj.strip()
        if stripped not in short_map and not is_placeholder(stripped):
            collected.add(stripped)


def apply_translations(obj, all_map):
    if isinstance(obj, dict):
        return {k: apply_translations(v, all_map) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [apply_translations(i, all_map) for i in obj]
    elif isinstance(obj, str):
        return clean_string(all_map.get(obj.strip(), obj))
    return obj


def translate_json(obj, short_map, tokenizer, model, lang_code):
    long_strings = set()
    collect_long_strings(obj, short_map, long_strings)
    long_map = batch_translate(long_strings, tokenizer, model, lang_code)
    all_map = {**short_map, **long_map}
    return apply_translations(obj, all_map)


def translate_file(file: Path, lang_code: str, tokenizer, model, output_dir: Path):
    output_path = output_dir / f"{lang_code}" / file.name
    if output_path.exists():
        return f"⏩ Skipped {file.name} → {lang_code} (exists)"

    with file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    short_strings = set()
    collect_short_strings(data, short_strings)
    short_map = batch_translate(short_strings, tokenizer, model, lang_code)
    translated = translate_json(data, short_map, tokenizer, model, lang_code)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        json.dump(translated, out, ensure_ascii=False, indent=2)

    torch.cuda.empty_cache()
    gc.collect()
    return f"✅ {file.name} → {lang_code}"


from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    model: str
    prefix: Optional[str] = None
    batch: int = BATCH_MAX
    src_lang: Optional[str] = None
    tgt_lang: Optional[str] = None
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    num_beams: Optional[int] = 4
    repetition_penalty: Optional[float] = 1.2
    early_stopping: Optional[bool] = True
    tokenizer_cls: Optional[type] = None
    fallback: Optional["ModelConfig"] = None

    def get_tokenizer(self):
        if self.tokenizer_cls:
            return self.tokenizer_cls
        if "nllb" in self.model:
            from transformers import NllbTokenizer
            return NllbTokenizer

        from transformers import AutoTokenizer
        return AutoTokenizer

    def generation_args(self, tokenizer) -> dict:
        args = {}
        if self.tgt_lang:
            args["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(self.tgt_lang)
        if self.max_length is not None:
            args["max_length"] = self.max_length
        if self.temperature is not None:
            args["temperature"] = self.temperature
        if self.top_p is not None:
            args["top_p"] = self.top_p
        if self.temperature is not None or self.top_p is not None:
            args["do_sample"] = True
        if self.num_beams is not None:
            args["num_beams"] = self.num_beams
        if self.repetition_penalty is not None:
            args["repetition_penalty"] = self.repetition_penalty
        if self.early_stopping is not None:
            args["early_stopping"] = self.early_stopping
        return args


from dataclasses import replace

FALLBACK_MODEL_CONFIGS: dict[str, ModelConfig] = {
    "bg": ModelConfig(
        model="Helsinki-NLP/opus-mt-tc-big-en-bg",
        prefix="bg",
        batch=48,
        num_beams=4,
        repetition_penalty=1.2,
        early_stopping=True
    ),
    "hu": ModelConfig(
        model="Helsinki-NLP/opus-mt-tc-big-en-hu",
        prefix="hu",
        batch=64,
        num_beams=5,
        repetition_penalty=1.3,
        early_stopping=True
    ),
    "sk": ModelConfig(
        model="allegro/multislav-5lang",
        prefix="slk",
        batch=4,
        num_beams=6,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.3,
        early_stopping=True
    ),
    "tr": ModelConfig(
        model="Helsinki-NLP/opus-mt-tc-big-en-tr",
        prefix="tr",
        batch=64,
        num_beams=5,
        repetition_penalty=1.2,
        early_stopping=True
    )
}

MODEL_CONFIGS: dict[str, ModelConfig] = {
    "bg": ModelConfig(
        model="facebook/nllb-200-distilled-600M",
        src_lang="eng_Latn",
        tgt_lang="bul_Cyrl",
        batch=8,
        max_length=256,
        temperature=0.8,
        top_p=0.95,
        num_beams=6,
        repetition_penalty=1.3,
        early_stopping=True,
        fallback=FALLBACK_MODEL_CONFIGS["bg"]
    ),
    "cs": ModelConfig(
        model="facebook/nllb-200-distilled-600M",
        src_lang="eng_Latn",
        tgt_lang="ces_Latn",
        batch=8,
        max_length=256,
        temperature=0.8,
        top_p=0.95,
        num_beams=6,
        repetition_penalty=1.3,
        early_stopping=True
    ),
    "hr": ModelConfig(
        model="facebook/nllb-200-distilled-600M",
        src_lang="eng_Latn",
        tgt_lang="hrv_Latn",
        batch=8,
        max_length=256,
        temperature=0.8,
        top_p=0.95,
        num_beams=4,
        repetition_penalty=1.2,
        early_stopping=True
    ),
    "hu": ModelConfig(
        model="facebook/nllb-200-distilled-600M",
        src_lang="eng_Latn",
        tgt_lang="hun_Latn",
        batch=8,
        max_length=256,
        temperature=0.8,
        top_p=0.95,
        num_beams=6,
        repetition_penalty=1.3,
        early_stopping=True,
        fallback=FALLBACK_MODEL_CONFIGS["hu"]
    ),
    "he": ModelConfig(
        model="facebook/nllb-200-distilled-600M",
        src_lang="eng_Latn",
        tgt_lang="heb_Hebr",
        batch=8,
        max_length=256,
        temperature=0.8,
        top_p=0.95,
        num_beams=6,
        repetition_penalty=1.3,
        early_stopping=True
    ),
    "lt": ModelConfig(
        model="facebook/nllb-200-distilled-600M",
        src_lang="eng_Latn",
        tgt_lang="lit_Latn",
        batch=8,
        max_length=256,
        temperature=0.8,
        top_p=0.95,
        num_beams=6,
        repetition_penalty=1.3,
        early_stopping=True
    ),
    "ro": ModelConfig(
        model="facebook/nllb-200-distilled-600M",
        src_lang="eng_Latn",
        tgt_lang="ron_Latn",
        batch=8,
        max_length=256,
        temperature=0.8,
        top_p=0.95,
        num_beams=6,
        repetition_penalty=1.3,
        early_stopping=True
    ),
    "sk": ModelConfig(
        model="facebook/nllb-200-distilled-600M",
        src_lang="eng_Latn",
        tgt_lang="slk_Latn",
        batch=8,
        max_length=256,
        temperature=0.8,
        top_p=0.95,
        num_beams=6,
        repetition_penalty=1.3,
        early_stopping=True,
        fallback=FALLBACK_MODEL_CONFIGS["sk"]
    ),
    "sl": ModelConfig(
        model="facebook/nllb-200-distilled-600M",
        src_lang="eng_Latn",
        tgt_lang="slv_Latn",
        batch=16,
        max_length=256,
        temperature=0.8,
        top_p=0.95,
        num_beams=6,
        repetition_penalty=1.3,
        early_stopping=True
    ),
    "tr": ModelConfig(
        model="facebook/nllb-200-distilled-600M",
        src_lang="eng_Latn",
        tgt_lang="tur_Latn",
        batch=8,
        max_length=256,
        temperature=0.8,
        top_p=0.95,
        num_beams=6,
        repetition_penalty=1.3,
        early_stopping=True,
        fallback=FALLBACK_MODEL_CONFIGS["tr"]
    )
}


def translate_language(lang_code: str, input_files: list[Path], output_dir: Path):
    from transformers import AutoModelForSeq2SeqLM
    base_cfg = MODEL_CONFIGS.get(lang_code)
    model_cfg = base_cfg

    try:
        tokenizer_cls = model_cfg.get_tokenizer()
        tokenizer = tokenizer_cls.from_pretrained(model_cfg.model)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_cfg.model).to(DEVICE)
        if model_cfg.src_lang:
            tokenizer.src_lang = model_cfg.src_lang

    except Exception as e:
        if base_cfg.fallback:
            print(f"⚠️ Falling back to {base_cfg.fallback.model}")
            model_cfg = replace(base_cfg.fallback, batch=base_cfg.batch)  # override batch or others if needed
            tokenizer_cls = model_cfg.get_tokenizer()
            tokenizer = tokenizer_cls.from_pretrained(model_cfg.model)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_cfg.model).to(DEVICE)
            if model_cfg.src_lang:
                tokenizer.src_lang = model_cfg.src_lang
        else:
            print(f"❌ Failed to load model {base_cfg.model}: {e}")
            return

    progress_bar = tqdm(input_files, desc=f"[{lang_code.upper()}]", position=TARGET_LANGS.index(lang_code), leave=False)
    for file in progress_bar:
        result = translate_file(file, lang_code, tokenizer, model, output_dir)
        progress_bar.write(result)

    del tokenizer, model
    torch.cuda.empty_cache()
    gc.collect()


def translate_all(input_dir: Path, output_dir: Path):
    input_files = list(input_dir.glob("*.json"))
    for lang_code in TARGET_LANGS:
        translate_language(lang_code, input_files, output_dir)


if __name__ == "__main__":
    input_dir = Path(r"C:\Users\Naliwe\Code\andragog\frontend\apps\frontend\src\locales\en")
    output_dir = Path(r"translated")
    translate_all(input_dir, output_dir)
