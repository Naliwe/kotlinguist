# Kotlinguist

This project reimplements a Python translation script in idiomatic
Kotlin/JVM (tested with Kotlin 2.2.0).  It relies on
[kotlinx.serialization](https://github.com/Kotlin/kotlinx.serialization)
for JSON handling and loads ONNX models exported from
[HuggingFace](https://huggingface.co/).

The program recursively translates strings in the source JSON files,
adding minimal context when necessary and reusing translations for
short strings. The translation logic is represented by an interface
`Translator`; `OnnxTranslator` loads an exported model with ONNX Runtime
and uses DJL's HuggingFace `tokenizers` library for preprocessing and decoding.

Run the program with:

```bash
./gradlew run --args="<inputDir> <outputDir>"
```

By default it looks for JSON files in `en/` and writes translated
output to `translated/`.

### 📦 Embedded Libraries

This project includes [ONNX Runtime](https://github.com/microsoft/onnxruntime), which is:

- Licensed under the [MIT License](https://github.com/microsoft/onnxruntime/blob/main/LICENSE)
- Copyright © Microsoft Corporation
- Bundled for convenience in `resources/native` (`onnxruntime.dll` / `libonnxruntime.so`)

A copy of the original license is included in [`LICENSE-ONNX.txt`](./src/main/resources/native/LICENSE-ONNX.txt).
