package pl.wtum.fingercounter

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import kotlin.math.max

class HandResult(val xs: FloatArray, val ys: FloatArray, val count: Int, val confidence: Float)

/** Wynik jednej klatki. Punkty są znormalizowane do pionowej klatki o wymiarach [frameWidth] x [frameHeight]. */
class FrameResult(
    val hands: List<HandResult>,
    val landmarkMs: Float,
    val mlpMs: Float,
    val frameWidth: Int,
    val frameHeight: Int,
)

/** Klatka z kamery -> MediaPipe (21 punktów na dłoń) -> wrist_scale -> MLP. */
class HandAnalyzer(context: Context, private val onResult: (FrameResult) -> Unit) : ImageAnalysis.Analyzer, AutoCloseable {
    private val landmarker: HandLandmarker
    private val classifier: FingerClassifier
    private val features = FloatArray(Preprocessing.NUM_FEATURES)

    // Bitmapy i macierz używane ponownie, żeby nie alokować pamięci na każdą klatkę
    private var source: Bitmap? = null
    private var upright: Bitmap? = null
    private var uprightCanvas: Canvas? = null
    private val rotation = Matrix()
    private var lastTimestamp = 0L

    init {
        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build())
            // VIDEO jest synchroniczny, więc czas MediaPipe i MLP mierzymy osobno na tej samej klatce
            .setRunningMode(RunningMode.VIDEO)
            .setNumHands(2)
            .build()
        landmarker = HandLandmarker.createFromOptions(context, options)
        classifier = FingerClassifier(context.assets.open("finger_counter.onnx").use { it.readBytes() })
    }

    override fun analyze(image: ImageProxy) {
        val frame = image.use { toUpright(it) }

        val t0 = SystemClock.elapsedRealtimeNanos()
        val timestamp = max(SystemClock.uptimeMillis(), lastTimestamp + 1).also { lastTimestamp = it }
        val detection = landmarker.detectForVideo(BitmapImageBuilder(frame).build(), timestamp)
        val t1 = SystemClock.elapsedRealtimeNanos()

        val hands = detection.landmarks().map { points ->
            val xs = FloatArray(Preprocessing.NUM_POINTS) { points[it].x() }
            val ys = FloatArray(Preprocessing.NUM_POINTS) { points[it].y() }
            val probs = classifier.predict(Preprocessing.wristScale(xs, ys, features))
            val count = FingerClassifier.argmax(probs)
            HandResult(xs, ys, count, probs[count])
        }
        val t2 = SystemClock.elapsedRealtimeNanos()

        onResult(FrameResult(hands, (t1 - t0) / 1e6f, (t2 - t1) / 1e6f, frame.width, frame.height))
    }

    /**
     * Obraca klatkę do pionu. Model uczył się na zdjęciach w pionie, a sensor daje obraz obrócony o 90 stopni.
     * Obracamy sami zamiast przez ImageProcessingOptions, żeby punkty na pewno były w układzie pionowej klatki.
     */
    private fun toUpright(image: ImageProxy): Bitmap {
        val w = image.width
        val h = image.height
        val plane = image.planes[0]
        val src = if (plane.rowStride == w * plane.pixelStride) {
            val bmp = source?.takeIf { it.width == w && it.height == h }
                ?: Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also { source = it }
            plane.buffer.rewind()
            bmp.copyPixelsFromBuffer(plane.buffer)
            bmp
        } else {
            image.toBitmap()
        }

        val degrees = image.imageInfo.rotationDegrees
        if (degrees == 0) return src

        val (uw, uh) = if (degrees % 180 == 0) w to h else h to w
        val dst = upright?.takeIf { it.width == uw && it.height == uh }
            ?: Bitmap.createBitmap(uw, uh, Bitmap.Config.ARGB_8888).also {
                upright = it
                uprightCanvas = Canvas(it)
            }
        rotation.reset()
        rotation.postTranslate(-w / 2f, -h / 2f)
        rotation.postRotate(degrees.toFloat())
        rotation.postTranslate(uw / 2f, uh / 2f)
        uprightCanvas!!.drawBitmap(src, rotation, null)
        return dst
    }

    override fun close() {
        landmarker.close()
        classifier.close()
    }
}
