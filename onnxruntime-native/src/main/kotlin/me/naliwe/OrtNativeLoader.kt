package me.naliwe

import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption

object OrtNativeLoader {
    private val os: String = System.getProperty("os.name").lowercase()
    private val arch: String = System.getProperty("os.arch")

    private val libExt: String = when {
        os.contains("win") -> "dll"
        os.contains("mac") -> "dylib"
        else -> "so"
    }

    private val classifier: String = when {
        os.contains("linux") && arch.contains("amd64") -> "linux-x86_64"
        os.contains("mac") && arch.contains("aarch64") -> "darwin-arm64"
        os.contains("mac") && arch.contains("x86_64") -> "darwin-x86_64"
        os.contains("win") && arch.contains("amd64") -> "windows-x86_64"
        else -> throw UnsupportedOperationException("Unsupported platform: $os $arch")
    }

    private var loaded = false

    fun load() {
        if (loaded) return

        val libName = when (libExt) {
            "dll" -> "onnxruntime.dll"
            "dylib" -> "libonnxruntime.dylib"
            else -> "libonnxruntime.so"
        }

        val resourcePath = "/native/$libName"
        val resource = OrtNativeLoader::class.java.getResourceAsStream(resourcePath)

        if (resource == null) {
            System.err.println("⚠️ Native ONNX library not found at $resourcePath, skipping load.")
            return
        }

        val tempFile = File.createTempFile("libonnxruntime", ".$libExt").apply {
            deleteOnExit()
        }

        Files.copy(resource, tempFile.toPath(), StandardCopyOption.REPLACE_EXISTING)
        System.load(tempFile.absolutePath)

        loaded = true
    }

    fun tryLoad(): Boolean {
        return try {
            load()
            true
        } catch (e: Exception) {
            System.err.println("⚠️ Failed to load ONNX native: ${e.message}")
            false
        }
    }
}
