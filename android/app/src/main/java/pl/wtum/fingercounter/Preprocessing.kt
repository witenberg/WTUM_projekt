package pl.wtum.fingercounter

import kotlin.math.max
import kotlin.math.sqrt

/** Preprocessing `wrist_scale` z notebooka 05, przepisany 1:1. */
object Preprocessing {
    const val NUM_POINTS = 21
    const val NUM_FEATURES = 2 * NUM_POINTS

    /**
     * Przesuwa punkty tak, żeby nadgarstek był w (0, 0), i dzieli przez odległość najdalszego punktu.
     * Wynik: x0, y0, x1, y1, ..., x20, y20.
     */
    fun wristScale(xs: FloatArray, ys: FloatArray, out: FloatArray = FloatArray(NUM_FEATURES)): FloatArray {
        val wx = xs[0]
        val wy = ys[0]
        var scale = 1e-6f
        for (i in 0 until NUM_POINTS) {
            val dx = xs[i] - wx
            val dy = ys[i] - wy
            out[2 * i] = dx
            out[2 * i + 1] = dy
            scale = max(scale, sqrt(dx * dx + dy * dy))
        }
        for (i in 0 until NUM_FEATURES) out[i] /= scale
        return out
    }
}
