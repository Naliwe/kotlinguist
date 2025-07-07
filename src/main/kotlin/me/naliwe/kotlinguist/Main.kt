package me.naliwe.kotlinguist

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.default
import kotlinx.serialization.json.*
import java.io.File

private const val MAX_SHORT_STRING_WORDS = 3

private val json = Json { prettyPrint = true }

fun contextualize(text: String): String =
    if (text.split(" ").size <= 1) "Label: $text" else "This label is: '$text'"

fun extractDecontextualized(translated: String): String {
    val parts = translated.split(":")

    return if (parts.size > 1) parts.last().trim().trim('"', '\'') else translated.trim()
}

fun collectShortStrings(node: JsonElement, shortStrings: MutableSet<String>) {
    when (node) {
        is JsonObject -> node.values.forEach { collectShortStrings(it, shortStrings) }
        is JsonArray -> node.forEach { collectShortStrings(it, shortStrings) }
        is JsonPrimitive -> if (node.isString && node.content.split(" ").size <= MAX_SHORT_STRING_WORDS) {
            shortStrings.add(node.content)
        }
    }
}

fun batchTranslate(strings: Set<String>, translator: Translator): Map<String, String> {
    if (strings.isEmpty()) return emptyMap()

    val sorted = strings.toList().sorted()
    val contextualized = sorted.map(::contextualize)
    val translated = translator.translate(contextualized)

    return sorted.zip(translated.map(::extractDecontextualized)).toMap()
}

fun translateJson(node: JsonElement, shortMap: Map<String, String>, translator: Translator): JsonElement = when (node) {
    is JsonObject -> JsonObject(node.mapValues { translateJson(it.value, shortMap, translator) })
    is JsonArray -> JsonArray(node.map { translateJson(it, shortMap, translator) })
    is JsonPrimitive -> if (node.isString) {
        val text = node.content
        val translated =
            shortMap[text] ?: translator.translate(listOf(contextualize(text)))[0].let(::extractDecontextualized)
        JsonPrimitive(translated)
    } else node
}

fun translateFile(file: File, langCode: String, translator: Translator, outputDir: File): String {
    val outputPath = File(outputDir, "$langCode-${langCode.uppercase()}/${file.name}")
    if (outputPath.exists()) return "⏩ Skipped ${file.name} → ${langCode} (exists)"

    val data = Json.parseToJsonElement(file.readText())
    val shortStrings = mutableSetOf<String>()
    collectShortStrings(data, shortStrings)

    val shortMap = batchTranslate(shortStrings, translator)
    val translated = translateJson(data, shortMap, translator)

    outputPath.parentFile.mkdirs()
    outputPath.writeText(json.encodeToString(JsonElement.serializer(), translated))

    return "✅ ${file.name} → ${langCode}"
}

fun main(args: Array<String>) {
    val parser = ArgParser("kotlinguist")
    val inputDirArg by parser.option(ArgType.String, shortName = "i", description = "Input directory").default("en")
    val outputDirArg by parser.option(ArgType.String, shortName = "o", description = "Output directory")
        .default("translated")
    parser.parse(args)

    val inputDir = File(inputDirArg)
    val outputDir = File(outputDirArg)
    if (!inputDir.exists()) {
        println("Input directory ${'$'}{inputDir.path} does not exist")
        return
    }

    val inputFiles = inputDir.listFiles { f -> f.extension == "json" }?.toList() ?: emptyList()
    val targetLangs = listOf("bg", "hr", "hu", "he", "lt", "ro", "sk", "sl", "tr", "cs")

    targetLangs.forEach { lang ->
        val modelDir = File("models/$lang")
        val translator = OnnxTranslator("${modelDir.path}/model.onnx", "${modelDir.path}/tokenizer.json")
        inputFiles.forEach { file ->
            val result = translateFile(file, lang, translator, outputDir)
            println(result)
        }
    }
}

