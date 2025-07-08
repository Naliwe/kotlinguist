import org.gradle.api.DefaultTask
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.model.ObjectFactory
import org.gradle.api.provider.Property
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.OutputFile
import org.gradle.api.tasks.TaskAction
import org.gradle.process.ExecOperations
import javax.inject.Inject

abstract class ExtractSoTask @Inject constructor(
    private val execOps: ExecOperations,
    objects: ObjectFactory
) : DefaultTask() {
    @get:Input
    val containerName: Property<String> = objects.property(String::class.java)

    @get:Input
    val image: Property<String> = objects.property(String::class.java)

    @get:OutputFile
    val output: RegularFileProperty = objects.fileProperty()

    @TaskAction
    fun extract() {
        val container = containerName.get()
        val target = output.get().asFile

        execOps.exec {
            commandLine("docker", "create", "--name", container, image.get())
        }

        execOps.exec {
            commandLine("docker", "cp", "$container:/build/libonnxruntime.so", target.absolutePath)
        }

        execOps.exec {
            commandLine("docker", "rm", container)
        }
    }
}
