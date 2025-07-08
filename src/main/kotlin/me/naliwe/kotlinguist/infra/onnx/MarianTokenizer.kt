package me.naliwe.kotlinguist.infra.onnx

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import me.naliwe.kotlinguist.domain.ports.Tokenizer
import java.nio.LongBuffer
import java.nio.file.Files
import java.nio.file.Paths

class MarianTokenizer(tokenizerPath: String) : Tokenizer {
    private val tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerPath))
    private val env = OrtEnvironment.getEnvironment()

    private val decoderStartTokenId: Long = run {
        val configPath = Paths.get(tokenizerPath).parent.resolve("config.json")

        if (Files.exists(configPath)) {
            val json = Json.parseToJsonElement(Files.readString(configPath)).jsonObject

            json["decoder_start_token_id"]?.jsonPrimitive?.intOrNull?.toLong() ?: 250004L
        } else 250004L
    }

    override fun encode(batch: List<String>): Map<String, OnnxTensor> {
        val inputIds = batch.map { tokenizer.encode(it).ids }
        val maxLen = inputIds.maxOf { it.size }
        val padded = inputIds.map { it + LongArray(maxLen - it.size) { 0 } }

        val flatInput = padded.flatMap { it.asList() }.toLongArray()
        val shape = longArrayOf(batch.size.toLong(), maxLen.toLong())
        val inputTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(flatInput), shape)

        val attentionMask = padded.map { row -> row.map { if (it == 0L) 0L else 1L }.toLongArray() }
        val flatMask = attentionMask.flatMap { it.asList() }.toLongArray()
        val maskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(flatMask), shape)

        val decoderStart = LongArray(batch.size) { decoderStartTokenId }
        val decoderTensor =
            OnnxTensor.createTensor(env, LongBuffer.wrap(decoderStart), longArrayOf(batch.size.toLong(), 1))

        return mapOf(
            "input_ids" to inputTensor,
            "attention_mask" to maskTensor,
            "decoder_input_ids" to decoderTensor
        )
    }
}
