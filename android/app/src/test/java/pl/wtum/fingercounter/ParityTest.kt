package pl.wtum.fingercounter

import org.json.JSONArray
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File

/** Sprawdza, że telefon liczy to samo co notebooki (deploy/test_vectors.json). */
class ParityTest {
    private val deploy = File("../../deploy")
    private val vectors = JSONArray(File(deploy, "test_vectors.json").readText())

    private fun JSONArray.toFloats() = FloatArray(length()) { getDouble(it).toFloat() }

    @Test
    fun preprocessingMatchesPython() {
        for (i in 0 until vectors.length()) {
            val v = vectors.getJSONObject(i)
            val raw = v.getJSONArray("raw_landmarks_xy")
            val xs = FloatArray(raw.length()) { raw.getJSONArray(it).getDouble(0).toFloat() }
            val ys = FloatArray(raw.length()) { raw.getJSONArray(it).getDouble(1).toFloat() }
            // Surowe punkty w JSON są zaokrąglone do 6 miejsc, po podzieleniu przez małą skalę błąd rośnie do ~1.5e-5
            assertArrayEquals("dłoń $i", v.getJSONArray("features").toFloats(), Preprocessing.wristScale(xs, ys), 5e-5f)
        }
    }

    @Test
    fun onnxMatchesPython() {
        FingerClassifier(File(deploy, "finger_counter.onnx").readBytes()).use { model ->
            for (i in 0 until vectors.length()) {
                val v = vectors.getJSONObject(i)
                val probs = model.predict(v.getJSONArray("features").toFloats())
                assertArrayEquals("dłoń $i", v.getJSONArray("expected_probabilities").toFloats(), probs, 1e-5f)
                assertEquals("dłoń $i", v.getInt("expected_label"), FingerClassifier.argmax(probs))
            }
        }
    }
}
