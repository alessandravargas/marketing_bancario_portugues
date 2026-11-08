"""Pré-processamento (porte fiel de `criar_preprocessadores` do notebook)."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

from . import config


def criar_preprocessadores():
    """
    Estratégia:
    1. Preprocessador Linear (Regressão Logística): discretiza 'pdays' via
       KBinsDiscretizer em faixas, evitando a distorção matemática do código -1
       em modelos lineares.
    2. Preprocessador de Árvores (Random Forest / XGBoost): mantém 'pdays'
       contínuo (com StandardScaler), pois árvores fazem splits nativos no -1.

    A coluna 'default' não é listada e, portanto, é descartada (remainder='drop').
    """
    num_cols_gerais = list(config.NUM_COLS_GERAIS)
    cat_cols = list(config.CAT_COLS)

    preproc_linear = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols_gerais),
            ("pdays_binned", KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="uniform"), ["pdays"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    num_cols_tree = num_cols_gerais + ["pdays"]
    preproc_tree = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols_tree),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    return preproc_linear, preproc_tree
