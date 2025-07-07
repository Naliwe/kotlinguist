package me.naliwe.kotlinguist

import ai.onnxruntime.*
import tokenizers.Tokenizer

/**
 * Simple translator backed by an ONNX model exported from HuggingFace.
 * Actual translation logic is not implemented and should be filled in
 * with sequence-to-sequence inference using the tokenizer and model.
 */
class OnnxTranslator(
    modelPath: String,
    tokenizerPath: String
) : Translator {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(modelPath, OrtSession.SessionOptions())
    private val tokenizer: Tokenizer = Tokenizer.fromFile(tokenizerPath)

    override fun translate(texts: List<String>): List<String> {
        val results = mutableListOf<String>()
        for (text in texts) {
            val enc = tokenizer.encode(text)
            val ids = enc.ids.map(Int::toLong).toLongArray()
            val mask = enc.attentionMask.map(Int::toLong).toLongArray()

            OnnxTensor.createTensor(env, longArrayOf(1, ids.size.toLong()), ids).use { idsTensor ->
                OnnxTensor.createTensor(env, longArrayOf(1, mask.size.toLong()), mask).use { maskTensor ->
                    val output = session.run(mapOf("input_ids" to idsTensor, "attention_mask" to maskTensor))
                    output.use {
                        val outIds = (it[0].value as Array<LongArray>)[0]
                        results.add(tokenizer.decode(outIds.map(Long::toInt).toLongArray(), true))
                    }
                }
            }
        }
        return results
    }
}
