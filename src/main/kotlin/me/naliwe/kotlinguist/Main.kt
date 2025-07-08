package me.naliwe.kotlinguist

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.required
import kotlinx.coroutines.runBlocking
import me.naliwe.kotlinguist.app.TranslationUseCase
import me.naliwe.kotlinguist.domain.ports.MarianTranslator
import me.naliwe.kotlinguist.infra.fs.ParallelFileWalker
import me.naliwe.kotlinguist.infra.onnx.loadMarianModel
import java.nio.file.Path

fun main(args: Array<String>): Unit = runBlocking {
    val parser = ArgParser("translate")
    val input by parser.option(ArgType.String, shortName = "i", description = "Input directory").required()
    val output by parser.option(ArgType.String, shortName = "o", description = "Output directory").required()
    val langs by parser.option(ArgType.String, shortName = "l", description = "Target languages (comma-separated)")
        .required()
    parser.parse(args)

    val useCase = TranslationUseCase(
        fileWalker = ParallelFileWalker(),
        translator = MarianTranslator(modelLoader = ::loadMarianModel)
    )

    useCase.execute(Path.of(input), Path.of(output), langs.split(",").map { it.trim() })
}
