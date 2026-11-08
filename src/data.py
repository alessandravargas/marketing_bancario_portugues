"""Carga dos dados.

Fonte: Kaggle `abdelazizsami/bank-marketing` (arquivo `bank-full.csv`),
derivado do UCI Bank Marketing (Moro, Cortez & Rita, 2014).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from . import config


def baixar_dataset() -> Path:
    """Baixa o dataset via kagglehub (como no notebook) e devolve a pasta."""
    import kagglehub  # import tardio: só necessário para o download

    return Path(kagglehub.dataset_download(config.KAGGLE_DATASET))


def localizar_csv(csv_path: str | os.PathLike | None = None) -> Path:
    """Resolve o caminho do CSV.

    Ordem: argumento explícito -> data/raw/bank-full.csv -> download via kagglehub.
    """
    if csv_path:
        return Path(csv_path)
    local = config.RAW_DIR / config.CSV_NAME
    if local.exists():
        return local
    return baixar_dataset() / config.CSV_NAME


def carregar_dados(csv_path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Lê o CSV (separador ';') exatamente como no notebook."""
    caminho = localizar_csv(csv_path)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    return pd.read_csv(caminho, delimiter=config.CSV_SEP)


def separar_X_y(df: pd.DataFrame):
    """X = todas as colunas menos 'y'; y = {'yes': 1, 'no': 0}."""
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET].map({"yes": 1, "no": 0})
    return X, y


def dividir_treino_teste(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):
    """Divisão estratificada 80/20 (random_state=42), como no notebook."""
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
