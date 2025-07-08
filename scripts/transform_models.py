import os
from pathlib import Path

from transformers import MarianMTModel
from transformers.onnx import export

MODEL_OVERRIDES = {
    "bg": "Helsinki-NLP/opus-mt-tc-big-en-bg",
    "cs": "Helsinki-NLP/opus-mt-en-cs",
    "hu": "Helsinki-NLP/opus-mt-tc-big-en-hu",
    "ro": "Helsinki-NLP/opus-mt-en-ro",
    "sk": "Helsinki-NLP/opus-mt-en-sk",
    "tr": "Helsinki-NLP/opus-mt-tc-big-en-tr",
    "hr": "Helsinki-NLP/opus-mt-en-mul",
    "he": "Helsinki-NLP/opus-mt-en-he",
    "lt": "Helsinki-NLP/opus-mt-tc-big-en-lt",
    "sl": "Helsinki-NLP/opus-mt-en-mul"
}

# These models need language token prepended to input
MULTILINGUAL_MODELS = {"hr", "sl"}

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")  # Set your token here if needed

output_root = Path("onnx_models")
output_root.mkdir(exist_ok=True)

for lang, model_name in MODEL_OVERRIDES.items():
    print(f"\n🔁 Exporting {model_name} for {lang.upper()} → {output_root / lang}")
    output_dir = output_root / lang
    output_dir.mkdir(parents=True, exist_ok=True)

    export(
        model=MarianMTModel.from_pretrained(model_name, use_auth_token=os.getenv("HF_TOKEN")),
        tokenizer=model_name,
        output=output_dir,
        feature="seq2seq-lm"
    )

    if lang in MULTILINGUAL_MODELS:
        print(f"⚠️  Note: {lang} uses a multilingual model. Be sure to prepend '>>{lang}<< ' to inputs.")

print("\n✅ All models exported to 'onnx_models/'")
