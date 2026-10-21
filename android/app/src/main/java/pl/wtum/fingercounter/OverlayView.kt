package pl.wtum.fingercounter

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

/** Rysuje punkty dłoni i liczbę palców na podglądzie kamery. */
class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    private var result: FrameResult? = null
    private var mirrored = false

    private val density = resources.displayMetrics.density
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(200, 255, 255, 255)
        strokeWidth = 3 * density
    }
    private val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(0, 200, 140) }
    private val badgePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(0, 200, 140) }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 36 * density
        textAlign = Paint.Align.CENTER
        isFakeBoldText = true
    }

    fun show(result: FrameResult, mirrored: Boolean) {
        this.result = result
        this.mirrored = mirrored
        invalidate()
    }

    fun clear() {
        result = null
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val r = result ?: return
        // Ten sam kadr co PreviewView w trybie FILL_CENTER: skalujemy do wypełnienia i centrujemy
        val scale = max(width.toFloat() / r.frameWidth, height.toFloat() / r.frameHeight)
        val offsetX = (width - r.frameWidth * scale) / 2
        val offsetY = (height - r.frameHeight * scale) / 2
        fun px(x: Float) = (x * r.frameWidth * scale + offsetX).let { if (mirrored) width - it else it }
        fun py(y: Float) = y * r.frameHeight * scale + offsetY

        for (hand in r.hands) {
            for ((a, b) in CONNECTIONS) {
                canvas.drawLine(px(hand.xs[a]), py(hand.ys[a]), px(hand.xs[b]), py(hand.ys[b]), linePaint)
            }
            for (i in hand.xs.indices) canvas.drawCircle(px(hand.xs[i]), py(hand.ys[i]), 4 * density, pointPaint)

            // Plakietka z wynikiem nad najwyższym punktem dłoni
            val cx = hand.xs.indices.sumOf { px(hand.xs[it]).toDouble() }.toFloat() / hand.xs.size
            val cy = max(py(hand.ys.min()) - 40 * density, 32 * density)
            canvas.drawCircle(cx, cy, 30 * density, badgePaint)
            canvas.drawText(hand.count.toString(), cx, cy - (labelPaint.ascent() + labelPaint.descent()) / 2, labelPaint)
        }
    }

    companion object {
        private val CONNECTIONS = listOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 4,
            0 to 5, 5 to 6, 6 to 7, 7 to 8,
            5 to 9, 9 to 10, 10 to 11, 11 to 12,
            9 to 13, 13 to 14, 14 to 15, 15 to 16,
            13 to 17, 0 to 17, 17 to 18, 18 to 19, 19 to 20,
        )
    }
}
