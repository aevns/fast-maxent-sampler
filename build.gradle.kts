import com.vanniktech.maven.publish.JavaLibrary
import com.vanniktech.maven.publish.JavadocJar

plugins {
    id("java-library")
    id("signing")
    id("com.vanniktech.maven.publish") version "0.34.0"
}

group = "io.github.aevns"
version = "0.2.0"

repositories {
    mavenCentral()
}

val jmh: SourceSet by sourceSets.creating {
    compileClasspath += sourceSets.main.get().output
    runtimeClasspath += sourceSets.main.get().output
    resources.srcDir("src/test/resources")
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:6.0.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")

    "jmhImplementation"("org.openjdk.jmh:jmh-core:1.37")
    "jmhAnnotationProcessor"("org.openjdk.jmh:jmh-generator-annprocess:1.37")
}

tasks.withType<JavaCompile> {
    options.release = 21
}

tasks.test {
    useJUnitPlatform()
}

tasks.javadoc {
    // Fail on any doc problem, including public API without a comment.
    (options as StandardJavadocDocletOptions).apply {
        addBooleanOption("Xdoclint:all", true)
        addBooleanOption("Werror", true)
    }
}

tasks.register<JavaExec>("jmh") {
    group = "verification"
    description = "Runs the JMH benchmarks; pass JMH options with --args, e.g. --args=\"-prof gc\"."
    classpath = jmh.runtimeClasspath
    mainClass = "org.openjdk.jmh.Main"
}

signing {
    useGpgCmd()
}

mavenPublishing {
    // Configured explicitly: the plugin's auto-detected setup adds -Xdoclint:none to every Javadoc task.
    configure(JavaLibrary(javadocJar = JavadocJar.Javadoc(), sourcesJar = true))
    publishToMavenCentral()
    signAllPublications()

    pom {
        name = "fast-maxent-sampler"
        description = "High-performance maximum-entropy sampling under certain moment constraints."
        inceptionYear = "2026"
        url = "https://github.com/aevns/fast-maxent-sampler"
        licenses {
            license {
                name = "MIT License"
                url = "https://opensource.org/licenses/MIT"
                distribution = "repo"
            }
        }
        developers {
            developer {
                id = "aevns"
                name = "aevns"
                url = "https://github.com/aevns"
            }
        }
        scm {
            url = "https://github.com/aevns/fast-maxent-sampler"
            connection = "scm:git:https://github.com/aevns/fast-maxent-sampler.git"
            developerConnection = "scm:git:ssh://git@github.com/aevns/fast-maxent-sampler.git"
        }
    }
}
