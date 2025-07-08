package me.naliwe.kotlinguist

import me.naliwe.OrtNativeLoader
import me.naliwe.kotlinguist.domain.ports.translate
import me.naliwe.kotlinguist.domain.translation.ContextHelper.contextualize
import me.naliwe.kotlinguist.domain.translation.ContextHelper.decontextualize
import me.naliwe.kotlinguist.infra.onnx.loadMarianModel

fun main() {
    val sample = listOf("Settings", "Open file", "Cancel", "Are you sure?")
    val contextualized = sample.map(::contextualize)

    if (!OrtNativeLoader.tryLoad()) {
        throw Exception("No OrtNativeLoader found")
    }

    val (session, tokenizer) = loadMarianModel("ro")
    val encoded = tokenizer.encode(contextualized)
    val outputs = session.translate(encoded)

    sample.zip(outputs).forEach { (orig, result) ->
        println("$orig → ${decontextualize(result, orig)}")
    }
}
