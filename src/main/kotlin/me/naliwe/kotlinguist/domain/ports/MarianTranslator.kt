package me.naliwe.kotlinguist.domain.ports

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import me.naliwe.kotlinguist.domain.translation.Batcher.adaptiveChunks
import me.naliwe.kotlinguist.domain.translation.ContextHelper.contextualize
import me.naliwe.kotlinguist.domain.translation.ContextHelper.decontextualize
import me.naliwe.kotlinguist.infra.fs.JsonUtils
import java.nio.file.Path
import kotlin.io.path.name

class MarianTranslator(
    private val modelLoader: (String) -> Pair<OrtSession, Tokenizer>
) : Translator {

    override suspend fun translateAll(inputFiles: List<Path>, targetLang: String, outputDir: Path) =
        withContext(Dispatchers.IO) {
            val (session, tokenizer) = modelLoader(targetLang)

            inputFiles.parallelStream().forEach { path ->
                val input = JsonUtils.parse(path)
                val shortStrings = JsonUtils.collectShortStrings(input)

                val shortTranslations = mutableMapOf<String, String>()
                for (batch in shortStrings.toList().adaptiveChunks()) {
                    val ctx = batch.map(::contextualize)
                    val tokens = tokenizer.encode(ctx)
                    val results = session.translate(tokens)
                    batch.zip(results).forEach { (orig, result) ->
                        shortTranslations[orig] = decontextualize(result, orig)
                    }
                }

                val replaced = JsonUtils.replaceStrings(input, shortTranslations)
                val outputPath = outputDir.resolve("$targetLang-${targetLang.uppercase()}/${path.name}")
                JsonUtils.write(outputPath, replaced)
            }
        }
}

interface Tokenizer {
    fun encode(batch: List<String>): Map<String, OnnxTensor>
}

fun OrtSession.translate(inputs: Map<String, OnnxTensor>): List<String> {
    return this.run(inputs).use { results ->
        @Suppress("UNCHECKED_CAST")
        results[0].value as List<String>
    }
}
