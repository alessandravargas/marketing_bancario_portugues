"""Executa o pipeline completo (equivale à célula 'EXECUÇÃO' do notebook).

Uso:
    python -m src.main                       # baixa via kagglehub (ou usa data/raw/bank-full.csv)
    python -m src.main --csv caminho.csv     # CSV local
    python -m src.main --skip-eda            # pula as figuras de EDA
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janela: figuras salvas em reports/figures

from . import config
from .data import carregar_dados, dividir_treino_teste, separar_X_y
from .eda import executar_eda
from .evaluation import avaliar_modelos_teste, plotar_avaliacao_comparativa
from .inference import InferenceService, salvar_modelo
from .interpretation import extrair_insights_modelo
from .models import treinar_modelos

# Cliente de exemplo idêntico ao do notebook (a chave 'default' é ignorada pelo pipeline).
NOVO_CLIENTE_EXEMPLO = {
    "age": 20, "balance": 500.0, "day": 15, "duration": 100, "campaign": 1, "pdays": 120,
    "previous": 2, "job": "management", "marital": "married", "education": "tertiary",
    "default": "no", "housing": "yes", "loan": "yes", "contact": "cellular", "month": "mar",
    "poutcome": "success",
}


def main(csv=None, skip_eda=False, save_model=False):
    df = carregar_dados(csv)
    print(f"Dados carregados: {df.shape[0]} linhas x {df.shape[1]} colunas")

    if not skip_eda:
        executar_eda(df)

    X, y = separar_X_y(df)
    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y)

    modelos = treinar_modelos(X_train, y_train, random_state=config.RANDOM_STATE)

    df_eval = avaliar_modelos_teste(modelos, X_test, y_test)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_eval.to_csv(config.REPORTS_DIR / "metricas_teste.csv", index=False)
    plotar_avaliacao_comparativa(modelos, X_test, y_test,
                                 save_path=config.FIGURES_DIR / "06_matrizes_confusao.png")

    best_model = modelos["XGBoost"]  # modelo final escolhido no notebook
    df_feat = extrair_insights_modelo(best_model)
    df_feat.to_csv(config.REPORTS_DIR / "top_features_xgboost.csv", index=False)

    print("=" * 75)
    print("ETAPA 7: TESTE DE INFERÊNCIA EM PRODUÇÃO")
    print("=" * 75)
    res = InferenceService(best_model).prever_propensao_cliente(NOVO_CLIENTE_EXEMPLO)
    print("Simulação de Inferência para Novo Cliente:")
    print(f"  - Probabilidade calculada: {res['probabilidade_conversao']*100:.2f}%")
    print(f"  - Ação recomendada:       {res['recomendacao_acao']}")
    print("=" * 75)
    (config.REPORTS_DIR / "exemplo_inferencia.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    if save_model:
        Path("models").mkdir(exist_ok=True)
        salvar_modelo(best_model, "models/xgboost_pipeline.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None, help="Caminho para bank-full.csv")
    ap.add_argument("--skip-eda", action="store_true")
    ap.add_argument("--save-model", action="store_true", help="Salva o pipeline XGBoost em models/")
    a = ap.parse_args()
    main(a.csv, a.skip_eda, a.save_model)
