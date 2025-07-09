import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    kotlin("jvm")
    id("com.github.johnrengelman.shadow") version "8.1.1"
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(project(":onnxruntime-native"))
    runtimeOnly(project(":onnxruntime-native:onnxruntime-binaries"))
}

tasks.named<ShadowJar>("shadowJar") {
    archiveClassifier.set("")
    mergeServiceFiles()
    manifest {
        attributes["Main-Class"] = "me.naliwe.onnxruntime.MainKt"
    }
}

tasks.build {
    dependsOn(tasks.shadowJar)
}
