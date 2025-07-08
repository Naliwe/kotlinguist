package me.naliwe.kotlinguist.domain.translation

object Batcher {
    fun <T> List<T>.adaptiveChunks(start: Int = 8, max: Int = 64): Sequence<List<T>> = sequence {
        var i = 0
        var batchSize = start

        while (i < size) {
            val end = minOf(i + batchSize, size)

            yield(subList(i, end))
            i = end

            if (batchSize < max) batchSize += 4
        }
    }
}
