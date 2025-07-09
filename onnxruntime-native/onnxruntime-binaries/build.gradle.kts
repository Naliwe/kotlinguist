plugins {
    `maven-publish`
    java
}

val targetClassifier = project.findProperty("nativeTarget")?.toString() ?: "linux-x86_64"
val enableCuda = project.findProperty("enableCuda") == "true"
val isWindows = targetClassifier.startsWith("windows")

val libExt = when {
    targetClassifier.startsWith("windows") -> "dll"
    targetClassifier.startsWith("darwin") -> "dylib"
    else -> "so"
}

val outputFile = layout.buildDirectory.file("libonnxruntime-$targetClassifier.$libExt")

val dockerFileName = "Dockerfile.$targetClassifier" + if (enableCuda) ".cuda" else ".cpu"
val dockerImage = "onnxruntime-builder:$targetClassifier" + if (enableCuda) "-cuda" else "-cpu"
val containerNameValue = "onnxruntime-tmp-$targetClassifier" + if (enableCuda) "-cuda" else "-cpu"

val buildDockerImage by tasks.registering(Exec::class) {
    group = "build"
    description = "Build Docker image for $targetClassifier ${if (enableCuda) "[CUDA]" else "[CPU]"}"

    commandLine("docker", "build", "-t", dockerImage, "-f", dockerFileName, ".")
    outputs.upToDateWhen { false }
}

val buildWindowsDll by tasks.registering(Exec::class) {
    group = "build"
    description = "Build onnxruntime.dll natively on Windows"

    onlyIf { isWindows && System.getProperty("os.name").startsWith("Windows") }

    workingDir = projectDir
    commandLine("cmd", "/c", "build-native.bat")

    outputs.file(outputFile)
}

val extractSo = tasks.register<ExtractSoTask>("extractSo") {
    dependsOn(buildDockerImage)
    onlyIf { !isWindows }
    image.set(dockerImage)
    containerName.set(containerNameValue)
    output.set(outputFile)
}

val nativeJar = tasks.register<Jar>("nativeJar") {
    group = "build"
    archiveClassifier.set(targetClassifier)

    from(outputFile) {
        into("native")
    }

    dependsOn(if (isWindows) buildWindowsDll else extractSo)
}

artifacts {
    add("runtimeElements", nativeJar)
}

publishing {
    publications {
        create<MavenPublication>("native") {
            artifactId = "onnxruntime-native"
            artifact(nativeJar) {
                classifier = targetClassifier
            }

            pom {
                name.set("ONNX Runtime Native")
                description.set("Native ONNX Runtime packaged for $targetClassifier")
                url.set("https://github.com/yourorg/yourrepo")
            }
        }
    }

    repositories {
        maven {
            name = "localMaven"
            url = uri("${rootProject.layout.buildDirectory}/maven-local")
        }
    }
}
