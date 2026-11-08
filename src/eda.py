"""Análise exploratória (porte das células de EDA do notebook).

Cada função devolve a figura e, opcionalmente, a salva em disco.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _salvar(fig, save_path):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def resumo_basico(df: pd.DataFrame) -> dict:
    """Tipos, nulos e estatísticas descritivas (df.info / df.describe)."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "nulos": df.isna().sum().to_dict(),
        "describe": df.describe(),
        "proporcao_target": df[config.TARGET].value_counts(normalize=True).to_dict(),
    }


def plot_target(df, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 4))
    total = len(df)
    sns.countplot(x="y", data=df, palette="viridis", hue="y", legend=False, ax=ax)
    ax.set_title("Distribuição da Variável Target", fontsize=12, fontweight="bold")
    ax.set_xlabel("Conversão(y)")
    ax.set_ylabel("Quantidade de Clientes")
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}\n({100 * h / total:.1f}%)", (p.get_x() + p.get_width() / 2., h),
                        ha="center", va="bottom", fontsize=10, fontweight="bold",
                        xytext=(0, 5), textcoords="offset points")
    ax.set_ylim(0, total * 1.1)
    fig.tight_layout()
    _salvar(fig, save_path)
    return fig


def plot_categoricas(df, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    cat_cols = df.select_dtypes(include=["object"]).columns
    n = len(cat_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten()
    total = len(df)
    for idx, col in enumerate(cat_cols):
        order = df[col].value_counts().index
        ax = sns.countplot(data=df, x=col, order=order, ax=axes[idx], palette="viridis", hue=col, legend=False)
        axes[idx].set_title(f"Distribuição de {col.upper()}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel("Frequência")
        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(f"{100 * h / total:.1f}%", (p.get_x() + p.get_width() / 2., h / 2.),
                            ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        if len(order) > 4 or df[col].astype(str).str.len().max() > 8:
            axes[idx].tick_params(axis="x", rotation=45)
            for lbl in axes[idx].get_xticklabels():
                lbl.set_ha("right")
    for idx in range(n, len(axes)):
        fig.delaxes(axes[idx])
    fig.tight_layout()
    _salvar(fig, save_path)
    return fig


def faixas_pdays_previous(df):
    """Cria 'Faixa pdays' e 'Faixa previous' (mesmos cortes do notebook)."""
    d = df.copy()
    d["Faixa pdays"] = np.select(
        [d["pdays"] == -1, (d["pdays"] > 0) & (d["pdays"] <= 90),
         (d["pdays"] > 90) & (d["pdays"] <= 180), d["pdays"] > 180],
        ["Nunca contatado", "Até 90 dias", "91 a 180 dias", "Mais de 180 dias"], default="Outros")
    d["Faixa previous"] = np.select(
        [d["previous"] == 0, (d["previous"] >= 1) & (d["previous"] <= 3), d["previous"] > 3],
        ["Sem contatos (0)", "Poucos contatos (1-3)", "Muitos contatos (>3)"], default="Outros")
    return d


def plot_pdays_previous_poutcome(df, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    d = faixas_pdays_previous(df)
    pivot = pd.crosstab(index=[d["Faixa pdays"], d["Faixa previous"]], columns=d["poutcome"],
                        normalize="index") * 100
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "Porcentagem (%)"},
                annot_kws={"weight": "bold", "size": 11}, ax=axes[0])
    axes[0].set_title("Sucesso do poutcome (%) por Perfil de Contato\n(Cruzamento de pdays + previous)",
                      fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Resultado da Campanha Anterior (poutcome)")
    axes[0].set_ylabel("Perfil (Faixa pdays, Faixa previous)")
    order = ["Nunca contatado", "Até 90 dias", "91 a 180 dias", "Mais de 180 dias"]
    sns.countplot(data=d, x="Faixa pdays", hue="poutcome", order=order, palette="viridis", ax=axes[1])
    axes[1].set_title("Volume de Clientes por Faixa de pdays e poutcome", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Faixas de Dias desde o Último Contato (pdays)")
    axes[1].set_ylabel("Quantidade de Clientes")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    _salvar(fig, save_path)
    return fig


def plot_pdays(df, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    status = (df["pdays"] == -1).map({True: "Nunca Contatados (pdays = -1)", False: "Já Contatados (pdays > 0)"})
    total = len(df)
    sns.countplot(x=status, palette="Set2", hue=status, legend=False, ax=axes[0])
    axes[0].set_title("Proporção de Clientes por Status de Contato Anterior", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Status de Contato")
    axes[0].set_ylabel("Quantidade de Clientes")
    for p in axes[0].patches:
        h = p.get_height()
        if h > 0:
            axes[0].annotate(f"{100 * h / total:.1f}%", (p.get_x() + p.get_width() / 2., h), ha="center",
                             va="bottom", fontsize=10, fontweight="bold", xytext=(0, 3),
                             textcoords="offset points")
    sns.histplot(data=df[df["pdays"] > 0], x="pdays", kde=True, color="teal", bins=30, ax=axes[1])
    axes[1].set_title("Distribuição de pdays (Apenas Clientes Contatados Anteriormente)",
                      fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Dias desde o último contato (pdays)")
    axes[1].set_ylabel("Frequência")
    fig.tight_layout()
    _salvar(fig, save_path)
    return fig


def associacao_com_alvo(df):
    """Correlação de Pearson (numéricas) e Informação Mútua (todas) com o alvo."""
    from sklearn.feature_selection import mutual_info_classif

    d = df.copy()
    d["target_bin"] = d["y"].map({"yes": 1, "no": 0})
    num_cols = d.select_dtypes(include=[np.number]).columns.drop("target_bin")
    corr = d[num_cols].corrwith(d["target_bin"]).sort_values(ascending=False)

    X_mi = d.drop(columns=["y", "target_bin"]).copy()
    for col in X_mi.select_dtypes(include=["object"]).columns:
        X_mi[col] = X_mi[col].astype("category").cat.codes
    mi = mutual_info_classif(X_mi, d["target_bin"], random_state=config.RANDOM_STATE)
    df_mi = (pd.DataFrame({"Variável": X_mi.columns, "Informação Mútua": mi})
             .sort_values(by="Informação Mútua", ascending=False))
    return corr, df_mi


def plot_associacao(df, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    corr, df_mi = associacao_com_alvo(df)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(x=corr.values, y=corr.index, ax=axes[0], palette="coolwarm", hue=corr.index, legend=False)
    axes[0].set_title("Correlação Linear de Pearson com o Alvo (Numéricas)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Coeficiente de Correlação")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    sns.barplot(x="Informação Mútua", y="Variável", data=df_mi, ax=axes[1], palette="viridis",
                hue="Variável", legend=False)
    axes[1].set_title("Dependência / Poder Preditivo (Mutual Information - Geral)", fontsize=12,
                      fontweight="bold")
    axes[1].set_xlabel("Score de Informação Mútua")
    fig.tight_layout()
    _salvar(fig, save_path)
    return fig, corr, df_mi


def executar_eda(df, out_dir=config.FIGURES_DIR):
    """Gera e salva todas as figuras de EDA."""
    out = Path(out_dir)
    plot_target(df, out / "01_target.png")
    plot_categoricas(df, out / "02_categoricas.png")
    plot_pdays_previous_poutcome(df, out / "03_pdays_previous_poutcome.png")
    plot_pdays(df, out / "04_pdays.png")
    _, corr, df_mi = plot_associacao(df, out / "05_associacao_alvo.png")
    return corr, df_mi
