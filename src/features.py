"""Funkcje pomocnicze do landmarków, wspólne dla notebooków."""
import numpy as np
import pandas as pd

from labels import FINGER_JOINTS, FINGERS

X_COLS = [f"x{i}" for i in range(21)]
Y_COLS = [f"y{i}" for i in range(21)]


def landmarks(df: pd.DataFrame) -> np.ndarray:
    """Zwraca tablicę (n, 21, 2) ze współrzędnymi landmarków znormalizowanymi do rozmiaru obrazu."""
    return np.stack([df[X_COLS].to_numpy(), df[Y_COLS].to_numpy()], axis=-1).astype(np.float32)


def normalise(lm: np.ndarray) -> np.ndarray:
    """Przesuwa nadgarstek do początku układu i skaluje dłoń tak, żeby mieściła się w kole jednostkowym."""
    lm = lm - lm[:, :1, :]
    scale = np.linalg.norm(lm, axis=-1).max(axis=1, keepdims=True)[..., None]
    return lm / np.maximum(scale, 1e-6)


def _angle(a, b, c):
    """Kąt w b (w stopniach) między odcinkami b->a i b->c, zwektoryzowany po pierwszej osi."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def extended_fingers(lm: np.ndarray, straight_deg: float = 150.0) -> pd.DataFrame:
    """Sprawdzenie regułą, które palce wyglądają na wyprostowane w 2D.

    Palec jest wyprostowany, gdy jego środkowe stawy są prawie proste, a czubek leży dalej
    od nadgarstka niż środkowy staw. To tylko kontrola mapowania etykiet, a nie model.
    """
    out = {}
    wrist = lm[:, 0]
    for name in FINGERS:
        j0, j1, j2, j3 = (lm[:, i] for i in FINGER_JOINTS[name])
        straight = (_angle(j0, j1, j2) > straight_deg) & (_angle(j1, j2, j3) > straight_deg)
        if name == "thumb":
            # Kciuk musi też odstawać od dłoni (czubek daleko od stawu MCP palca wskazującego).
            away = np.linalg.norm(j3 - lm[:, 5], axis=-1) > np.linalg.norm(j1 - lm[:, 5], axis=-1)
        else:
            away = np.linalg.norm(j3 - wrist, axis=-1) > np.linalg.norm(j1 - wrist, axis=-1)
        out[name] = straight & away
    df = pd.DataFrame(out)
    df["n_extended"] = df[FINGERS].sum(axis=1)
    return df
