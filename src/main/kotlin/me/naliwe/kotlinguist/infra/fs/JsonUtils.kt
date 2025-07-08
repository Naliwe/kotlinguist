package me.naliwe.kotlinguist.infra.fs

import kotlinx.serialization.json.*
import java.nio.file.Files
import java.nio.file.Path

object JsonUtils {
    private val json = Json { ignoreUnknownKeys = true; prettyPrint = true }

    fun parse(path: Path): JsonElement =
        Files.newBufferedReader(path).use { json.parseToJsonElement(it.readText()) }

    fun write(path: Path, data: JsonElement) {
        Files.createDirectories(path.parent)
        Files.newBufferedWriter(path).use {
            it.write(json.encodeToString(JsonElement.serializer(), data))
        }
    }

    fun collectShortStrings(json: JsonElement, maxWords: Int = 3): Set<String> = buildSet {
        fun recurse(element: JsonElement) {
            when (element) {
                is JsonObject -> element.values.forEach(::recurse)
                is JsonArray -> element.forEach(::recurse)
                is JsonPrimitive -> if (element.isString) {
                    val value = element.content
                    if (value.split(" ").size <= maxWords) add(value)
                }
            }
        }
        recurse(json)
    }

    fun replaceStrings(json: JsonElement, replacements: Map<String, String>): JsonElement {
        return when (json) {
            is JsonObject -> JsonObject(json.mapValues { (_, v) -> replaceStrings(v, replacements) })
            is JsonArray -> JsonArray(json.map { replaceStrings(it, replacements) })
            is JsonPrimitive -> if (json.isString) {
                val content = json.content
                JsonPrimitive(replacements[content] ?: content)
            } else json
        }
    }
}
