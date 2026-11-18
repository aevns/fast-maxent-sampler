plugins {
    id("java-library")
    id("signing")
    id("com.vanniktech.maven.publish") version "0.34.0"
}

group = "io.github.aevns"
version = "0.1.0"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:6.0.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<JavaCompile> {
    options.release = 21
}

tasks.test {
    useJUnitPlatform()
}

signing {
    useGpgCmd()
}

mavenPublishing {
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
