package me.naliwe.kotlinguist.app

import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import me.naliwe.kotlinguist.domain.ports.Translator
import me.naliwe.kotlinguist.infra.fs.FileWalker
import java.nio.file.Path

class TranslationUseCase(
    private val fileWalker: FileWalker,
    private val translator: Translator
) {
    suspend fun execute(inputDir: Path, outputDir: Path, targetLangs: List<String>) = coroutineScope {
        val files = fileWalker.listJsonFiles(inputDir)

        targetLangs.map { lang ->
            async {
                translator.translateAll(files, lang, outputDir)
            }
        }.awaitAll()
    }
}
