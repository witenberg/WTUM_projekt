plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "pl.wtum.fingercounter"
    compileSdk = 35

    defaultConfig {
        applicationId = "pl.wtum.fingercounter"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        // Tylko 64-bitowy ARM: tak działa praktycznie każdy telefon, a APK jest o połowę mniejszy
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Klucz debug wystarcza do instalacji poza Sklepem Play
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    androidResources { noCompress += listOf("task", "onnx") }
}

// Testy JVM używają desktopowego ONNX Runtime zamiast androidowego (ta sama paczka ai.onnxruntime)
configurations.matching { it.name.endsWith("UnitTestRuntimeClasspath") }.configureEach {
    exclude(group = "com.microsoft.onnxruntime", module = "onnxruntime-android")
}

dependencies {
    val camerax = "1.4.1"
    val onnxruntime = "1.20.0"

    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")
    implementation("com.google.mediapipe:tasks-vision:0.10.21")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:$onnxruntime")

    testImplementation("junit:junit:4.13.2")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:$onnxruntime")
    testImplementation("org.json:json:20240303")
}
