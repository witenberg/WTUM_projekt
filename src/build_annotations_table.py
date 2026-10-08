"""Spłaszcza adnotacje JSON z HaGRIDv2 do dwóch tabel parquet.

images.parquet: jeden wiersz na obraz (podział, klasa źródłowa, osoba, liczba dłoni, metadane głównej osoby).
hands.parquet:  jeden wiersz na dłoń z adnotacją (etykieta, ramka, 21 landmarków MediaPipe jako x0..x20, y0..y20).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data/raw/annotations"
OUT = ROOT / "data/interim"


def first(meta, key):
    vals = (meta or {}).get(key) or []
    return vals[0] if vals else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images, hands = [], []
    lm_cols = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
    for path in tqdm(sorted(ANN.glob("*/*.json"))):
        split, cls = path.parent.name, path.stem
        for image_id, rec in json.load(open(path)).items():
            labels = rec["labels"] or []
            images.append({
                "image_id": image_id, "split": split, "source_class": cls, "user_id": rec["user_id"],
                "n_hands": len(labels), "n_gesture_hands": sum(l != "no_gesture" for l in labels),
                "has_united": rec.get("united_label") is not None,
                "age": first(rec.get("meta"), "age"), "gender": first(rec.get("meta"), "gender"),
                "race": first(rec.get("meta"), "race"), "n_people": len((rec.get("meta") or {}).get("age") or []),
            })
            lms = rec.get("hand_landmarks") or [[] for _ in labels]
            for h, (label, bbox, lm) in enumerate(zip(labels, rec["bboxes"], lms)):
                row = {"image_id": image_id, "split": split, "source_class": cls, "user_id": rec["user_id"],
                       "hand_idx": h, "label": label, "bx": bbox[0], "by": bbox[1], "bw": bbox[2], "bh": bbox[3],
                       "has_landmarks": len(lm) == 21}
                arr = np.asarray(lm, dtype=np.float32) if len(lm) == 21 else np.full((21, 2), np.nan, np.float32)
                row.update(dict(zip(lm_cols, np.concatenate([arr[:, 0], arr[:, 1]]))))
                hands.append(row)
    pd.DataFrame(images).to_parquet(OUT / "images.parquet", index=False)
    pd.DataFrame(hands).to_parquet(OUT / "hands.parquet", index=False)
    print(len(images), "obrazów,", len(hands), "dłoni")


if __name__ == "__main__":
    main()
