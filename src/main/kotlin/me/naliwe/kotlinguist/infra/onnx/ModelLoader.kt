package me.naliwe.kotlinguist.infra.onnx

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import me.naliwe.kotlinguist.domain.ports.Tokenizer
import java.nio.file.Path

val modelOverrides = mapOf(
    "bg" to "models/opus-mt-tc-big-en-bg",
    "cs" to "models/opus-mt-en-cs",
    "hu" to "models/opus-mt-tc-big-en-hu",
    "ro" to "models/opus-mt-en-ro",
    "sk" to "models/opus-mt-en-sk",
    "tr" to "models/opus-mt-tc-big-en-tr",
    "hr" to "models/opus-mt-en-mul",
    "he" to "models/opus-mt-en-he",
    "lt" to "models/opus-mt-tc-big-en-lt",
    "sl" to "models/opus-mt-en-mul"
)

fun loadMarianModel(lang: String): Pair<OrtSession, Tokenizer> {
    val modelPath = modelOverrides[lang] ?: "models/opus-mt-en-$lang"
    val env = OrtEnvironment.getEnvironment()
    val session = env.createSession(Path.of("$modelPath/model.onnx").toString(), SessionOptions())
    val tokenizer = MarianTokenizer("$modelPath/tokenizer.json")

    return session to tokenizer
}
