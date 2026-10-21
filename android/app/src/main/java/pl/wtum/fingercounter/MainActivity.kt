package pl.wtum.fingercounter

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.View
import android.view.WindowManager
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var countText: TextView
    private lateinit var statsText: TextView
    private lateinit var startScreen: View
    private lateinit var cameraScreen: View

    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private var analyzer: HandAnalyzer? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var lensFacing = CameraSelector.LENS_FACING_FRONT

    private var lastFrameNs = 0L
    private var fps = 0f

    private val requestCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else Toast.makeText(this, R.string.no_permission, Toast.LENGTH_LONG).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        // Ekran nie gaśnie w trakcie nagrywania filmu
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        previewView = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        countText = findViewById(R.id.count)
        statsText = findViewById(R.id.stats)
        startScreen = findViewById(R.id.start_screen)
        cameraScreen = findViewById(R.id.camera_screen)

        findViewById<View>(R.id.start_button).setOnClickListener {
            val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
            if (granted) startCamera() else requestCamera.launch(Manifest.permission.CAMERA)
        }
        findViewById<View>(R.id.switch_button).setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) CameraSelector.LENS_FACING_BACK
            else CameraSelector.LENS_FACING_FRONT
            overlay.clear()
            bindCamera()
        }
    }

    private fun startCamera() {
        startScreen.visibility = View.GONE
        cameraScreen.visibility = View.VISIBLE
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            cameraProvider = future.get()
            bindCamera()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCamera() {
        val provider = cameraProvider ?: return
        // 16:9 jak zdjęcia HaGRID (1080x1920), żeby proporcje znormalizowanych punktów były te same co w treningu
        val ratio = AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY
        val preview = Preview.Builder()
            .setResolutionSelector(ResolutionSelector.Builder().setAspectRatioStrategy(ratio).build())
            .build()
        preview.setSurfaceProvider(previewView.surfaceProvider)

        // MediaPipe i tak skaluje obraz do 192-224 px, więc mała klatka wystarcza i oszczędza czas
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setAspectRatioStrategy(ratio)
                    .setResolutionStrategy(ResolutionStrategy(Size(640, 360), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                    .build()
            )
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        val handAnalyzer = analyzer ?: HandAnalyzer(this, ::onFrame).also { analyzer = it }
        analysis.setAnalyzer(executor, handAnalyzer)

        val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
        try {
            provider.unbindAll()
            provider.bindToLifecycle(this, selector, preview, analysis)
        } catch (e: Exception) {
            Log.e(TAG, "Nie udało się uruchomić kamery", e)
            Toast.makeText(this, R.string.camera_error, Toast.LENGTH_LONG).show()
        }
    }

    /** Wywoływane na wątku analizy. */
    private fun onFrame(result: FrameResult) {
        val now = SystemClock.elapsedRealtimeNanos()
        if (lastFrameNs != 0L) {
            val instant = 1e9f / (now - lastFrameNs)
            fps = if (fps == 0f) instant else 0.9f * fps + 0.1f * instant
        }
        lastFrameNs = now

        val counts = result.hands.map { it.count }
        val label = when (counts.size) {
            0 -> getString(R.string.no_hand)
            1 -> counts[0].toString()
            else -> counts.joinToString(" + ") + " = " + counts.sum()
        }
        val stats = getString(R.string.stats, result.landmarkMs, result.mlpMs, fps)
        val mirrored = lensFacing == CameraSelector.LENS_FACING_FRONT

        runOnUiThread {
            countText.text = label
            statsText.text = stats
            overlay.show(result, mirrored)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraProvider?.unbindAll()
        // Zamykamy modele na wątku analizy, żeby nie zamknąć ich w trakcie przetwarzania klatki
        executor.execute { analyzer?.close() }
        executor.shutdown()
    }

    companion object {
        private const val TAG = "FingerCounter"
    }
}
