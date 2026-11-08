"""Constantes do projeto (valores idênticos aos do notebook original)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"

KAGGLE_DATASET = "abdelazizsami/bank-marketing"
CSV_NAME = "bank-full.csv"
CSV_SEP = ";"

TARGET = "y"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Colunas usadas na modelagem. 'default' NÃO é usada (descartada pelo
# ColumnTransformer, remainder='drop', como no notebook).
NUM_COLS_GERAIS = ["age", "balance", "day", "duration", "campaign", "previous"]
NUM_COLS_TREE = NUM_COLS_GERAIS + ["pdays"]
CAT_COLS = ["job", "marital", "education", "housing", "loan", "contact", "month", "poutcome"]

# Custos de negócio do notebook (PREMISSAS do autor do notebook, não dados reais).
CUSTO_FP = 15    # R$ – custo da ligação (tempo de atendente)
CUSTO_FN = 150   # R$ – contrato perdido

# Limiar de decisão usado no InferenceService do notebook.
LIMIAR_DECISAO = 0.5

# Grade de hiperparâmetros do XGBoost (idêntica ao notebook).
PARAM_GRID_XGB = {
    "clf__n_estimators": [50, 100],
    "clf__max_depth": [3, 5],
    "clf__learning_rate": [0.05, 0.1],
}
CV_FOLDS = 3
CV_SCORING = "roc_auc"
