"""Pomiar GreenAI i wspólny dziennik eksperymentów używany przez bramki procesu."""
import csv
import datetime as dt
import time
from contextlib import contextmanager
from pathlib import Path

from codecarbon import OfflineEmissionsTracker

LOG = Path(__file__).resolve().parents[1] / "experiments/log.csv"
FIELDS = ["date", "experiment_id", "path", "process_step", "model", "train_samples", "seeds", "val_accuracy",
          "val_macro_f1", "val_macro_f1_std", "val_stress_macro_f1", "val_stress_macro_f1_std", "train_time_s",
          "energy_kwh", "co2_g", "model_size_kb", "latency_ms", "gate", "notes"]


@contextmanager
def track(name: str):
    """Mierzy czas, energię i CO2 bloku kodu. Intensywność emisji sieci w Polsce, bez połączeń sieciowych."""
    tracker = OfflineEmissionsTracker(project_name=name, country_iso_code="POL", log_level="critical",
                                      save_to_file=False, measure_power_secs=5)
    result = {}
    tracker.start()
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result["train_time_s"] = time.perf_counter() - t0
        result["co2_g"] = (tracker.stop() or 0.0) * 1000
        result["energy_kwh"] = tracker.final_emissions_data.energy_consumed


def log_experiment(**row):
    """Dopisuje jeden wiersz do experiments/log.csv i zastępuje starszy wiersz o tym samym experiment_id."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if LOG.exists():
        with open(LOG) as f:
            rows = [r for r in csv.DictReader(f) if r["experiment_id"] != row["experiment_id"]]
    row = {"date": dt.date.today().isoformat(), **row}
    rows.append({k: row.get(k, "") for k in FIELDS})
    with open(LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
