package me.naliwe.kotlinguist

import ai.onnxruntime.*
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer

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
    private val tokenizer: HuggingFaceTokenizer = HuggingFaceTokenizer.newInstance(tokenizerPath)

    override fun translate(texts: List<String>): List<String> {
        val results = mutableListOf<String>()
        for (text in texts) {
            val enc = tokenizer.encode(text)
            val ids = enc.ids
            val mask = enc.attentionMask

            OnnxTensor.createTensor(env, arrayOf(ids)).use { idsTensor ->
                OnnxTensor.createTensor(env, arrayOf(mask)).use { maskTensor ->
                    val output = session.run(mapOf("input_ids" to idsTensor, "attention_mask" to maskTensor))
                    output.use {
                        val outIds = (it[0].value as Array<LongArray>)[0]
                        results.add(tokenizer.decode(outIds, true))
                    }
                }
            }
        }
        return results
    }
}
