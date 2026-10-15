# Liczenie palców w czasie rzeczywistym (WTUM)

Aplikacja na Androida, która na żywo liczy wyprostowane palce z obrazu kamery. Model trenowany na HaGRIDv2.
Proces i wymagania kursu są w `CLAUDE.md` i `docs/proces_projektowy.png`.

## Struktura

| Ścieżka | Zawartość |
|---|---|
| `notebooks/01_dataset_review.ipynb` | Krok procesu 1. Wybór zbioru danych, HaGRID v1 vs v2, mapowanie gestu na liczbę palców |
| `notebooks/02_eda.ipynb` | EDA na wszystkich 1,086,158 adnotacjach. Balans, klasy, jakość, sprawdzenie mapowania |
| `notebooks/03_path_keypoints.ipynb` | Eksploracja ścieżki A. Klasyfikator na landmarkach MediaPipe, z metrykami GreenAI |
| `notebooks/04_baseline.ipynb` | Krok 2. Baseline (MLP na surowych punktach), zasady oceny, test obciążeniowy i reguła bramek |
| `notebooks/05_preprocessing.ipynb` | Kroki 3-4. Warianty preprocessingu punktów i bramka. Opisana błędna ścieżka z obrotem do pionu |
| `notebooks/06_augmentation.ipynb` | Kroki 5-6. Augmentacja punktów (obrót, proporcje, odbicie, szum) i bramka |
| `notebooks/07_tuning.ipynb` | Kroki 7-8. Random search hiperparametrów MLP i bramka |
| `notebooks/08_final_model.ipynb` | Krok 9. Wynik na zbiorze testowym, gesty, grupy osób, GreenAI całego procesu |
| `notebooks/09_xai.ipynb` | Krok 10. SHAP: które punkty dłoni decydują o liczbie palców, przykłady pomyłek |
| `notebooks/10_export.ipynb` | Krok 11 (część ML). Eksport do ONNX, zgodność ze sklearn, wektory testowe dla aplikacji |
| `deploy/` | Model ONNX, specyfikacja wejść i wyjść, wektory testowe i opis pipeline'u na Androidzie |
| `experiments/gates.json` | Decyzje bramek. Każdy notebook czyta wynik poprzedniego kroku stąd |
| `experiments/log.csv` | Jeden wiersz na eksperyment, metryki jakości i GreenAI. Źródło danych dla bramek procesu |
| `reports/figures/` | Wykresy zapisane przez notebooki, gotowe do dokumentacji i slajdów |
| `src/` | Wspólny kod. Spłaszczanie adnotacji, mapowanie etykiet, cechy, pomiar GreenAI |
| `data/` | Poza repozytorium. `raw/` to pobrane archiwa, `interim/` to tabele parquet |
| `models/` | Poza repozytorium. Modele sklearn z każdego eksperymentu |

## Odtworzenie wyników

```bash
uv sync
curl -L -o data/raw/annotations.zip \
  https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip
unzip data/raw/annotations.zip -d data/raw
uv run python src/build_annotations_table.py
cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace 0*.ipynb 10_export.ipynb
```

Liczby GreenAI pochodzą z CodeCarbona w trybie offline, z intensywnością emisji polskiej sieci energetycznej.
Na macOS bez `sudo` CodeCarbon szacuje moc CPU na podstawie TDP procesora, więc wartości energii to szacunki,
a nie odczyty z miernika.
