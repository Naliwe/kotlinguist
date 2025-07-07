plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
    application
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(libs.kotlin.stdlib)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.kotlinx.cli)

    implementation(libs.onnxruntime)
    implementation(libs.tokenizers)
}

application {
    mainClass.set("me.naliwe.kotlinguist.MainKt")
}
