package me.naliwe.kotlinguist.domain.ports

import java.nio.file.Path

fun interface Translator {
    suspend fun translateAll(inputFiles: List<Path>, targetLang: String, outputDir: Path)
}
