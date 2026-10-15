# Wdrożenie modelu na Androida

Ten folder zawiera wszystko, czego aplikacja potrzebuje od strony ML. Pliki generuje `notebooks/10_export.ipynb`.

| Plik | Co to jest |
|---|---|
| `finger_counter.onnx` | Model końcowy (StandardScaler + MLP) w formacie ONNX |
| `model_spec.json` | Nazwy i kształty wejść i wyjść, kolejność cech, wyniki na teście |
| `test_vectors.json` | 18 dłoni z testu: surowe punkty, cechy po preprocessingu, oczekiwane prawdopodobieństwa |

## Pipeline na telefonie (na każdą klatkę kamery)

1. **CameraX** `ImageAnalysis` daje klatkę.
2. **MediaPipe Hand Landmarker** (`com.google.mediapipe:tasks-vision`, model `hand_landmarker.task`,
   tryb `LIVE_STREAM`, `numHands = 2`) zwraca 21 punktów na dłoń, `x` i `y` znormalizowane do szerokości
   i wysokości obrazu.
   Trzeba podać obrót klatki (`ImageProcessingOptions.setRotationDegrees(imageProxy.imageInfo.rotationDegrees)`).
   Bez tego punkty są w układzie sensora, czyli obrócone o 90 stopni, a model zna tylko przechylenia do około
   30-45 stopni.
3. **Preprocessing** (`wrist_scale`), dokładnie jak w treningu:
   ```kotlin
   // lm: 21 par (x, y) z MediaPipe
   val wx = lm[0].x; val wy = lm[0].y
   val dx = FloatArray(21) { lm[it].x - wx }
   val dy = FloatArray(21) { lm[it].y - wy }
   val scale = max(1e-6f, (0 until 21).maxOf { sqrt(dx[it] * dx[it] + dy[it] * dy[it]) })
   // Kolejność cech: x0, y0, x1, y1, ..., x20, y20 (przeplecione)
   val features = FloatArray(42) { i -> if (i % 2 == 0) dx[i / 2] / scale else dy[i / 2] / scale }
   ```
4. **ONNX Runtime** (`com.microsoft.onnxruntime:onnxruntime-android`). Wejście `float32 [1, 42]`, wyjścia
   `label` (int64) i `probabilities` (float32 [1, 6], klasy 0-5 palców). Nazwy są w `model_spec.json`.
5. Wynik nakładamy na podgląd kamery. Bez dłoni pokazujemy "brak dłoni".

## Sprawdzenie zgodności w aplikacji

Test jednostkowy w aplikacji powinien wczytać `test_vectors.json` i dla każdej dłoni sprawdzić dwie rzeczy:
- preprocessing z `raw_landmarks_xy` daje `features` (tolerancja około 1e-5),
- ONNX Runtime na `features` daje `expected_probabilities` (tolerancja około 1e-5) i `expected_label`.

Jeśli oba testy przechodzą, telefon liczy dokładnie to samo co notebooki.

## Czego model nie zna

- Dłoni obróconych mocniej niż około 45 stopni (stąd punkt 2 o obrocie klatki).
- Układów palców spoza 25 gestów HaGRID, np. "3" pokazane jako serdeczny + mały + środkowy. To trzeba sprawdzić
  na własnych nagraniach (weryfikacja filmem jest i tak wymagana).
