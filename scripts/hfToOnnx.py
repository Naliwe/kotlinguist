import onnx
import torch
from argparse import ArgumentParser
from onnxconverter_common.float16 import convert_float_to_float16
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer
from transformers.utils import logging

logging.set_verbosity_error()

model_overrides = {
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


def export_fp16(model_id: str, output_dir: Path):
    print(f"🚀 Exporting {model_id}")
    model = MarianMTModel.from_pretrained(model_id)
    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)

    dummy = tokenizer("Translate this.", return_tensors="pt", padding="max_length", max_length=32)

    fp32_path = output_dir / "model-fp32.onnx"

    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    # Create dummy decoder_input_ids (Marian expects BOS token)
    decoder_input_ids = torch.full((1, 1), model.config.decoder_start_token_id, dtype=torch.long)

    torch.onnx.export(
        model,
        args=(input_ids, attention_mask, decoder_input_ids),
        f=fp32_path.as_posix(),
        input_names=["input_ids", "attention_mask", "decoder_input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "decoder_input_ids": {0: "batch_size", 1: "decoder_sequence_length"}
        },
        opset_version=14
    )

    print("🔁 Converting to FP16...")
    model_fp32 = onnx.load(fp32_path)
    model_fp16 = convert_float_to_float16(model_fp32)
    onnx.save_model(model_fp16, output_dir / "model.onnx")
    fp32_path.unlink()

    print(f"✅ Done: {output_dir / 'model.onnx'}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_root = Path(args.output)
    for lang, model_id in model_overrides.items():
        export_fp16(model_id, output_root / f"opus-mt-{lang}")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()
