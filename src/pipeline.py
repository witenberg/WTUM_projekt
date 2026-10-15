"""Wspólny pipeline ścieżki A: dane, preprocessing, augmentacja, test obciążeniowy, trening z pomiarem GreenAI."""
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from features import landmarks, normalise
from greenai import log_experiment, track
from labels import FINGER_JOINTS, GESTURE_TO_COUNT

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SEEDS = (0, 1, 2)


def load_split(split: str):
    """Surowe landmarki (n, 21, 2) we współrzędnych obrazu, etykiety 0-5 i ramka z metadanymi dłoni."""
    cols = ["split", "label", "has_landmarks", "image_id"] + [f"{a}{i}" for a in "xy" for i in range(21)]
    hands = pd.read_parquet(ROOT / "data/interim/hands.parquet", columns=cols)
    d = hands[(hands.split == split) & hands.label.isin(list(GESTURE_TO_COUNT)) & hands.has_landmarks].reset_index(drop=True)
    return landmarks(d), d.label.map(GESTURE_TO_COUNT).to_numpy(), d


# --- Transformacje geometryczne (wspólne dla augmentacji i testu obciążeniowego) ---

def _hand_scale(lm):
    return np.linalg.norm(lm - lm.mean(1, keepdims=True), axis=-1).max(1)


def transform(lm, rng, max_rot_deg, aspect_range, p_mirror, jitter):
    """Losowo obraca dłoń, zmienia proporcje osi x/y, odbija lustrzanie i dodaje szum. Wszystko wokół środka dłoni,
    więc położenie dłoni w kadrze się nie zmienia."""
    n = len(lm)
    centre = lm.mean(1, keepdims=True)
    out = lm - centre
    # Odbicie lustrzane to lewa dłoń zamiast prawej albo kamera przednia zamiast tylnej.
    mirror = rng.random(n) < p_mirror
    out[mirror, :, 0] *= -1
    # Inne proporcje obrazu rozciągają współrzędne znormalizowane do szerokości i wysokości kadru.
    out[:, :, 0] *= rng.uniform(*aspect_range, n)[:, None]
    theta = np.radians(rng.uniform(-max_rot_deg, max_rot_deg, n))
    c, s = np.cos(theta), np.sin(theta)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    out = np.einsum("nij,nkj->nki", rot, out)
    out += rng.normal(0, 1, out.shape) * (jitter * _hand_scale(lm))[:, None, None]
    return (out + centre).astype(np.float32)


def stress_set(lm, seed=123):
    """Stała, trudniejsza kopia zbioru: przechylona dłoń (do 45 stopni), proporcje 4:3 do 16:9, połowa odbita,
    lekki szum detektora. Symuluje warunki telefonu, których HaGRID prawie nie ma."""
    return transform(lm, np.random.default_rng(seed), max_rot_deg=45, aspect_range=(0.75, 1.33), p_mirror=0.5,
                     jitter=0.01)


STRESS_PARTS = {
    "tylko obrót": dict(max_rot_deg=45, aspect_range=(1, 1), p_mirror=0, jitter=0),
    "tylko proporcje": dict(max_rot_deg=0, aspect_range=(0.75, 1.33), p_mirror=0, jitter=0),
    "tylko odbicie": dict(max_rot_deg=0, aspect_range=(1, 1), p_mirror=0.5, jitter=0),
    "tylko szum": dict(max_rot_deg=0, aspect_range=(1, 1), p_mirror=0, jitter=0.01),
}


def augment(lm, y, rng, copies=1, max_rot_deg=30, aspect_range=(0.8, 1.25), p_mirror=0.5, jitter=0.01):
    """Dokłada `copies` losowo przekształconych kopii zbioru treningowego. Oryginał zostaje."""
    extra = [transform(lm, rng, max_rot_deg, aspect_range, p_mirror, jitter) for _ in range(copies)]
    return np.concatenate([lm, *extra]), np.concatenate([y] * (copies + 1))


# --- Warianty preprocessingu: (n, 21, 2) surowe -> (n, k) cechy ---

def _upright(lm):
    """Obraca dłoń tak, żeby wektor nadgarstek -> staw MCP palca środkowego wskazywał w górę."""
    v = lm[:, 9] - lm[:, 0]
    # Kąt wektora względem pionu (0, -1); obracamy o minus ten kąt, żeby go wyzerować.
    theta = -np.arctan2(v[:, 0], -v[:, 1])
    c, s = np.cos(theta), np.sin(theta)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    return np.einsum("nij,nkj->nki", rot, lm)


def _joint_angles(lm):
    """Kąty zgięcia w 3 stawach każdego palca (15 liczb, w radianach / pi)."""
    feats = []
    for joints in FINGER_JOINTS.values():
        chain = (0,) + joints
        for a, b, c in zip(chain, chain[1:], chain[2:]):
            v1, v2 = lm[:, a] - lm[:, b], lm[:, c] - lm[:, b]
            cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
            feats.append(np.arccos(np.clip(cos, -1, 1)) / np.pi)
    return np.stack(feats, 1)


PREPROCESS = {
    "raw": lambda lm: lm.reshape(len(lm), -1),
    "wrist_scale": lambda lm: normalise(lm).reshape(len(lm), -1),
    "wrist_scale_upright": lambda lm: _upright(normalise(lm)).reshape(len(lm), -1),
    "wrist_scale_angles": lambda lm: np.concatenate([normalise(lm).reshape(len(lm), -1), _joint_angles(lm)], 1),
    "wrist_scale_upright_angles": lambda lm: np.concatenate(
        [_upright(normalise(lm)).reshape(len(lm), -1), _joint_angles(lm)], 1),
}


# --- Trening, ewaluacja i wpis do dziennika ---

def macro_f1(model, X, y):
    return f1_score(y, model.predict(X), average="macro")


def latency_ms(model, x, n=500, repeats=5):
    """Czas predykcji jednej dłoni. Minimum z kilku serii, żeby obciążenie laptopa nie zaszumiało pomiaru."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n):
            model.predict_proba(x)
        times.append((time.perf_counter() - t0) / n * 1000)
    return min(times)


def run_experiment(exp_id, step, make_model, X_tr, y_tr, X_va, y_va, X_st, notes, seeds=SEEDS, gate=""):
    """Trenuje model dla kilku ziaren, mierzy GreenAI, zapisuje model z ziarna 0 i średnie wyniki do log.csv."""
    runs = []
    for seed in seeds:
        model = make_model(seed)
        with track(exp_id) as g:
            model.fit(X_tr, y_tr)
        runs.append({"seed": seed, "val_macro_f1": macro_f1(model, X_va, y_va),
                     "val_stress_macro_f1": macro_f1(model, X_st, y_va), **g})
        if seed == seeds[0]:
            MODELS.mkdir(exist_ok=True)
            blob = pickle.dumps(model)
            (MODELS / f"{exp_id}.pkl").write_bytes(blob)
            size_kb, lat = len(blob) / 1024, latency_ms(model, X_va[:1])
    r = pd.DataFrame(runs)
    summary = {
        "experiment_id": exp_id, "process_step": step, "train_samples": len(y_tr),
        "val_macro_f1": r.val_macro_f1.mean(), "val_macro_f1_std": r.val_macro_f1.std(),
        "val_stress_macro_f1": r.val_stress_macro_f1.mean(), "val_stress_macro_f1_std": r.val_stress_macro_f1.std(),
        "train_time_s": r.train_time_s.mean(), "energy_kwh": r.energy_kwh.mean(), "co2_g": r.co2_g.mean(),
        "model_size_kb": size_kb, "latency_ms": lat,
    }
    log_experiment(path="A keypoints", model=notes.get("model", ""), seeds=len(seeds), gate=gate,
                   notes=notes.get("notes", ""), **{k: (round(v, 6) if isinstance(v, float) else v)
                                                    for k, v in summary.items()})
    return summary, r


def gate(prev: dict, new: dict, k_std: float = 2.0) -> tuple[bool, list[str]]:
    """Bramka z diagramu. Zwraca decyzję i listę powodów (zasady opisane w 04_baseline)."""
    reasons = []
    noise = k_std * max(prev["val_stress_macro_f1_std"], new["val_stress_macro_f1_std"], 1e-4)
    d_stress = new["val_stress_macro_f1"] - prev["val_stress_macro_f1"]
    val_noise = k_std * max(prev["val_macro_f1_std"], new["val_macro_f1_std"], 1e-4)
    d_val = new["val_macro_f1"] - prev["val_macro_f1"]
    quality = d_stress > noise and d_val > -val_noise
    reasons.append(f"stress F1 {d_stress:+.4f} (próg szumu {noise:.4f}), val F1 {d_val:+.4f} (próg {val_noise:.4f})")
    # Rozmiar i opóźnienie mierzymy dla jednego modelu, więc dajemy 10% luzu na rozrzut pomiaru.
    use_cost = new["model_size_kb"] <= prev["model_size_kb"] * 1.1 and new["latency_ms"] <= prev["latency_ms"] * 1.1
    reasons.append(f"rozmiar {prev['model_size_kb']:.0f} -> {new['model_size_kb']:.0f} KB, "
                   f"predykcja {prev['latency_ms']:.3f} -> {new['latency_ms']:.3f} ms")
    reasons.append(f"energia treningu {prev['energy_kwh'] * 1000:.2f} -> {new['energy_kwh'] * 1000:.2f} Wh")
    return quality and use_cost, reasons


def feature_names(prep_name: str) -> list[str]:
    """Nazwy cech w kolejności kolumn: x0, y0, x1, y1, ... (punkty są przeplecione), potem ewentualne kąty."""
    names = [f"{a}{i}" for i in range(21) for a in "xy"]
    if prep_name.endswith("angles"):
        names += [f"kąt_{f}_{k}" for f in FINGER_JOINTS for k in (1, 2, 3)]
    return names
