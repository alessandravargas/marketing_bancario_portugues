import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.data import dividir_treino_teste, separar_X_y
from src.evaluation import avaliar_modelos_teste
from src.inference import InferenceService
from src.preprocessing import criar_preprocessadores


def _lr_pipeline(X_train, y_train):
    preproc_linear, _ = criar_preprocessadores()
    pipe = Pipeline([("preproc", preproc_linear),
                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    return pipe.fit(X_train, y_train)


def test_separar_e_dividir(df_sintetico):
    X, y = separar_X_y(df_sintetico)
    assert "y" not in X.columns and set(y.unique()) <= {0, 1}
    Xtr, Xte, ytr, yte = dividir_treino_teste(X, y)
    assert len(Xte) == pytest.approx(0.2 * len(X), abs=1)
    assert ytr.mean() == pytest.approx(yte.mean(), abs=0.01)  # estratificação


def test_preprocessadores_formas(df_sintetico):
    X, _ = separar_X_y(df_sintetico)
    lin, tree = criar_preprocessadores()
    Xl, Xt = lin.fit_transform(X), tree.fit_transform(X)
    # linear tem 4 colunas a mais (pdays em 4 faixas) e 1 a menos (pdays contínuo) => +3
    assert Xl.shape[1] == Xt.shape[1] + 3
    assert not np.isnan(Xl).any() and not np.isnan(Xt).any()


def test_default_e_descartada(df_sintetico):
    X, _ = separar_X_y(df_sintetico)
    _, tree = criar_preprocessadores()
    tree.fit(X)
    nomes = tree.get_feature_names_out()
    assert not any("default" in n for n in nomes)


def test_avaliacao_e_custo(df_sintetico):
    X, y = separar_X_y(df_sintetico)
    Xtr, Xte, ytr, yte = dividir_treino_teste(X, y)
    modelo = _lr_pipeline(Xtr, ytr)
    df_res = avaliar_modelos_teste({"LR": modelo}, Xte, yte, verbose=False)
    r = df_res.iloc[0]
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(yte, modelo.predict(Xte)).ravel()
    assert r["Custo Negócio (R$)"] == fp * config.CUSTO_FP + fn * config.CUSTO_FN
    assert 0 <= r["ROC-AUC"] <= 1 and 0 <= r["PR-AUC"] <= 1


def test_inference_service(df_sintetico):
    X, y = separar_X_y(df_sintetico)
    modelo = _lr_pipeline(X, y)
    cliente = X.iloc[0].to_dict()
    res = InferenceService(modelo).prever_propensao_cliente(cliente)
    assert 0 <= res["probabilidade_conversao"] <= 1
    esperado = ("Contatar Cliente (Alta Propensão)" if res["probabilidade_conversao"] >= 0.5
                else "Não Contatar (Baixa Propensão)")
    assert res["recomendacao_acao"] == esperado


def test_treino_completo_e_insights(df_sintetico):
    """Exige xgboost e imbalanced-learn (ignorado se ausentes)."""
    pytest.importorskip("xgboost")
    pytest.importorskip("imblearn")
    from src.interpretation import extrair_insights_modelo
    from src.models import treinar_modelos

    X, y = separar_X_y(df_sintetico)
    Xtr, Xte, ytr, yte = dividir_treino_teste(X, y)
    modelos = treinar_modelos(Xtr, ytr, verbose=False)
    assert set(modelos) == {"Regressão Logística", "Random Forest", "XGBoost"}
    top = extrair_insights_modelo(modelos["XGBoost"], verbose=False)
    assert len(top) == 8 and top["Importancia"].is_monotonic_decreasing


def test_eda_associacao(df_sintetico):
    from src.eda import associacao_com_alvo, faixas_pdays_previous
    corr, mi = associacao_com_alvo(df_sintetico)
    assert "duration" in corr.index and len(mi) == 16
    assert "Outros" not in set(faixas_pdays_previous(df_sintetico)["Faixa pdays"])


def test_plot_matrizes_confusao(df_sintetico, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from src.evaluation import plotar_avaliacao_comparativa
    X, y = separar_X_y(df_sintetico)
    Xtr, Xte, ytr, yte = dividir_treino_teste(X, y)
    modelo = _lr_pipeline(Xtr, ytr)
    out = tmp_path / "cm.png"
    plotar_avaliacao_comparativa({"LR": modelo}, Xte, yte, save_path=out)
    assert out.exists() and out.stat().st_size > 0
