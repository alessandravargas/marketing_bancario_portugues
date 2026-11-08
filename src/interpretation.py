"""Interpretação (porte fiel de `extrair_insights_modelo`)."""
import pandas as pd

from . import config


def extrair_insights_modelo(best_model, num_cols=None, cat_cols=None, top_n=8, verbose=True):
    """Importância das variáveis do classificador final (ex.: XGBoost).

    Devolve um DataFrame com as `top_n` variáveis (após one-hot) mais importantes.
    """
    num_cols = list(num_cols or config.NUM_COLS_TREE)
    cat_cols = list(cat_cols or config.CAT_COLS)

    clf = best_model.named_steps["clf"]
    preproc = best_model.named_steps["preproc"]

    cat_encoder = preproc.named_transformers_["cat"]
    feature_names = num_cols + list(cat_encoder.get_feature_names_out(cat_cols))

    if not hasattr(clf, "feature_importances_"):
        return pd.DataFrame(columns=["Atributo", "Importancia"])

    df_feat = (pd.DataFrame({"Atributo": feature_names, "Importancia": clf.feature_importances_})
               .sort_values(by="Importancia", ascending=False).head(top_n).reset_index(drop=True))

    if verbose:
        print("=" * 75)
        print("ETAPA 6: INTERPRETAÇÃO E INSIGHTS ACIONÁVEIS")
        print("=" * 75)
        print(f"Top {top_n} Atributos Mais Influentes na Decisão do Cliente:")
        for idx, row in df_feat.iterrows():
            print(f"  {idx+1}. {row['Atributo']:<30} (Peso: {row['Importancia']:.4f})")
        print("=" * 75 + "\n")
    return df_feat
