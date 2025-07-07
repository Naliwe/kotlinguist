# Kotlinguist

This project reimplements a Python translation script in idiomatic
Kotlin/JVM (tested with Kotlin 1.9.22).  It relies on
[kotlinx.serialization](https://github.com/Kotlin/kotlinx.serialization)
for JSON handling and loads ONNX models exported from
[HuggingFace](https://huggingface.co/).

The program recursively translates strings in the source JSON files,
adding minimal context when necessary and reusing translations for
short strings. The translation logic is represented by an interface
`Translator`; `OnnxTranslator` loads an exported model with ONNX Runtime
and uses HuggingFace `tokenizers` for preprocessing and decoding.

Run the program with:

```bash
./gradlew run --args="<inputDir> <outputDir>"
```

By default it looks for JSON files in `en/` and writes translated
output to `translated/`.
