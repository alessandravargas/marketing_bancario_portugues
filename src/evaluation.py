"""Avaliação (porte fiel de `avaliar_modelos_teste` e `plotar_avaliacao_comparativa`)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (auc, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from . import config


def avaliar_modelos_teste(modelos, X_test, y_test, custo_fp=config.CUSTO_FP,
                          custo_fn=config.CUSTO_FN, verbose=True):
    """
    Avalia os modelos no teste e calcula o impacto financeiro.
    custo = FP * custo_fp + FN * custo_fn  (premissas do notebook: R$15 e R$150).
    Ordenado por PR-AUC (decrescente), como no notebook.
    """
    resultados = []
    for nome, model in modelos.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        roc = roc_auc_score(y_test, y_prob)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(rec_curve, prec_curve)
        f1 = f1_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        custo_total = (fp * custo_fp) + (fn * custo_fn)

        resultados.append({
            "Modelo": nome,
            "ROC-AUC": round(roc, 4),
            "PR-AUC": round(pr_auc, 4),
            "F1-Score": round(f1, 4),
            "Recall": round(rec, 4),
            "Precisão": round(prec, 4),
            "Custo Negócio (R$)": custo_total,
        })

    df_res = pd.DataFrame(resultados).sort_values(by="PR-AUC", ascending=False)
    if verbose:
        print("=" * 75)
        print("ETAPA 5: AVALIAÇÃO NO CONJUNTO DE TESTE")
        print("=" * 75)
        print(df_res.to_string(index=False))
        print("=" * 75 + "\n")
    return df_res


def plotar_avaliacao_comparativa(modelos, X_test, y_test, custo_fp=config.CUSTO_FP,
                                 custo_fn=config.CUSTO_FN, save_path: str | Path | None = None,
                                 show: bool = False):
    """Matrizes de confusão com contagem, % por linha real e custo em R$ por quadrante."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    num_modelos = len(modelos)
    fig, axes = plt.subplots(1, num_modelos, figsize=(4.5 * num_modelos, 5))
    if num_modelos == 1:
        axes = [axes]
    cmaps = ["Blues", "Oranges", "Greens", "Purples", "Reds"]

    for idx, (nome, model) in enumerate(modelos.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        custo_tn = 0.0
        custo_fp_celula = fp * custo_fp
        custo_fn_celula = fn * custo_fn
        custo_tp = 0.0

        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
        labels = np.array([
            [f"{tn}\n({cm_percent[0,0]:.1f}%)\nR$ {custo_tn:,.2f}",
             f"{fp}\n({cm_percent[0,1]:.1f}%)\nR$ {custo_fp_celula:,.2f}"],
            [f"{fn}\n({cm_percent[1,0]:.1f}%)\nR$ {custo_fn_celula:,.2f}",
             f"{tp}\n({cm_percent[1,1]:.1f}%)\nR$ {custo_tp:,.2f}"],
        ])

        ax = axes[idx]
        sns.heatmap(cm, annot=labels, fmt="", cmap=cmaps[idx % len(cmaps)], ax=ax, cbar=False,
                    square=True, annot_kws={"size": 10, "weight": "bold"})
        ax.set_title(f"{nome}\nCusto Total: R$ {custo_fp_celula + custo_fn_celula:,.2f}",
                     fontsize=10, fontweight="bold", pad=10)
        ax.set_xlabel("Predito", fontsize=9)
        ax.set_ylabel("Real", fontsize=9)
        ax.set_xticklabels(["Não (0)", "Sim (1)"], fontsize=9)
        ax.set_yticklabels(["Não (0)", "Sim (1)"], fontsize=9, rotation=0)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
