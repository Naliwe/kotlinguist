import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    kotlin("jvm")
    id("com.github.johnrengelman.shadow") version "8.1.1"
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(project(":onnxruntime-native"))                         // includes OrtNativeLoader
    runtimeOnly(project(":onnxruntime-native:onnxruntime-binaries"))       // includes .so/.dll
}

tasks.named<ShadowJar>("shadowJar") {
    archiveClassifier.set("")  // no '-all' suffix
    mergeServiceFiles()        // if needed for SPI
    manifest {
        attributes["Main-Class"] = "com.yourcompany.onnxruntime.MainKt" // or any class with `main()`
    }
}

tasks.build {
    dependsOn(tasks.shadowJar)
}
