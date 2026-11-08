import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def df_sintetico():
    """Dados SINTÉTICOS (apenas para testar o código; não representam o dataset real)."""
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({
        "age": rng.integers(18, 90, n),
        "job": rng.choice(["admin.", "management", "technician", "unknown"], n),
        "marital": rng.choice(["married", "single", "divorced"], n),
        "education": rng.choice(["primary", "secondary", "tertiary", "unknown"], n),
        "default": rng.choice(["no", "yes"], n, p=[.9, .1]),
        "balance": rng.integers(-500, 20000, n),
        "housing": rng.choice(["yes", "no"], n),
        "loan": rng.choice(["yes", "no"], n),
        "contact": rng.choice(["cellular", "telephone", "unknown"], n),
        "day": rng.integers(1, 32, n),
        "month": rng.choice(["jan", "mar", "may", "jun", "oct"], n),
        "duration": rng.integers(0, 1500, n),
        "campaign": rng.integers(1, 20, n),
        "pdays": np.where(rng.random(n) < .8, -1, rng.integers(1, 800, n)),
        "previous": rng.integers(0, 10, n),
        "poutcome": rng.choice(["unknown", "failure", "success", "other"], n),
    })
    score = (df["duration"] / 1500) + (df["poutcome"] == "success") * 1.0 + rng.normal(0, .5, n)
    df["y"] = np.where(score > np.quantile(score, .85), "yes", "no")
    return df
