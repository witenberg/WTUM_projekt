package pl.wtum.fingercounter

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

/** Model końcowy (StandardScaler + MLP) uruchamiany przez ONNX Runtime. */
class FingerClassifier(modelBytes: ByteArray) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val options = OrtSession.SessionOptions().apply {
            // MLP 128-64 jest tak mały, że dodatkowe wątki tylko dokładają narzutu
            setIntraOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(modelBytes, options)
    }

    /** Zwraca prawdopodobieństwa klas 0-5 palców. */
    fun predict(features: FloatArray): FloatArray {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(features), longArrayOf(1, Preprocessing.NUM_FEATURES.toLong()))
            .use { input ->
                session.run(mapOf(INPUT to input), setOf(OUTPUT)).use { result ->
                    @Suppress("UNCHECKED_CAST")
                    return (result.get(0).value as Array<FloatArray>)[0]
                }
            }
    }

    override fun close() = session.close()

    companion object {
        const val INPUT = "X"
        const val OUTPUT = "probabilities"

        fun argmax(p: FloatArray): Int = p.indices.maxBy { p[it] }
    }
}
