package me.naliwe.kotlinguist.domain.translation

object ContextHelper {
    fun contextualize(input: String): String =
        if (input.split(" ").size <= 1) "Label: $input" else "This label is: '$input'"

    fun decontextualize(translated: String, original: String): String {
        val result = translated
            .substringAfter(":")
            .removeSurrounding("\"", "\"").trim('"', '\'', ' ')

        return result.ifBlank { original }
    }
}
