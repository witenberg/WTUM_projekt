# Liczenie palców w czasie rzeczywistym (WTUM)

Aplikacja na Androida, która na żywo liczy wyprostowane palce z obrazu kamery. Model trenowany na HaGRIDv2.
Proces i wymagania kursu są w `CLAUDE.md` i `docs/proces_projektowy.png`.

## Struktura

| Ścieżka | Zawartość |
|---|---|
| `notebooks/01_dataset_review.ipynb` | Krok procesu 1. Wybór zbioru danych, HaGRID v1 vs v2, mapowanie gestu na liczbę palców |
| `notebooks/02_eda.ipynb` | EDA na wszystkich 1,086,158 adnotacjach. Balans, klasy, jakość, sprawdzenie mapowania |
| `notebooks/03_path_keypoints.ipynb` | Eksploracja ścieżki A. Klasyfikator na landmarkach MediaPipe, z metrykami GreenAI |
| `experiments/log.csv` | Jeden wiersz na eksperyment, metryki jakości i GreenAI. Źródło danych dla bramek procesu |
| `reports/figures/` | Wykresy zapisane przez notebooki, gotowe do dokumentacji i slajdów |
| `src/` | Wspólny kod. Spłaszczanie adnotacji, mapowanie etykiet, cechy, pomiar GreenAI |
| `data/` | Poza repozytorium. `raw/` to pobrane archiwa, `interim/` to tabele parquet |
| `models/` | Poza repozytorium. Zapisane modele |

## Odtworzenie wyników

```bash
uv sync
curl -L -o data/raw/annotations.zip \
  https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip
unzip data/raw/annotations.zip -d data/raw
uv run python src/build_annotations_table.py
cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

Liczby GreenAI pochodzą z CodeCarbona w trybie offline, z intensywnością emisji polskiej sieci energetycznej.
Na macOS bez `sudo` CodeCarbon szacuje moc CPU na podstawie TDP procesora, więc wartości energii to szacunki,
a nie odczyty z miernika.
