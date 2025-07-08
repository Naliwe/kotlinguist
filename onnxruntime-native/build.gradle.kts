plugins {
    kotlin("jvm")
    `maven-publish`
}

repositories {
    mavenCentral()
}

dependencies {
    runtimeOnly(project(":onnxruntime-native:onnxruntime-binaries"))
}

publishing {
    publications {
        create<MavenPublication>("wrapper") {
            artifactId = "onnxruntime-native"
            from(components["java"])

            pom {
                name.set("ONNX Runtime Native Loader")
                description.set("Auto-loading ONNX Runtime native bindings")
            }
        }
    }
}
