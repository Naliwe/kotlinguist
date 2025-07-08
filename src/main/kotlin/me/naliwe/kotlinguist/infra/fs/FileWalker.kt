package me.naliwe.kotlinguist.infra.fs

import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.isRegularFile
import kotlin.io.path.name

interface FileWalker {
    fun listJsonFiles(inputDir: Path): List<Path>
}

class ParallelFileWalker : FileWalker {
    override fun listJsonFiles(inputDir: Path): List<Path> =
        Files.walk(inputDir)
            .parallel()
            .filter { it.isRegularFile() && it.name.endsWith(".json") }
            .toList()
}
