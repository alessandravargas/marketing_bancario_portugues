"""Construção e otimização dos modelos (porte fiel de `treinar_modelos`)."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from . import config
from .preprocessing import criar_preprocessadores


def treinar_modelos(X_train, y_train, random_state=config.RANDOM_STATE, verbose=True):
    """
    Treina cada modelo com o pipeline de pré-processamento específico.
    Balanceamento cost-sensitive (class_weight / scale_pos_weight); SMOTE NÃO é usado.
    """
    # Imports tardios: xgboost e imbalanced-learn só são exigidos ao treinar.
    from imblearn.pipeline import Pipeline as ImbPipeline
    from xgboost import XGBClassifier

    def log(msg):
        if verbose:
            print(msg)

    log("=" * 75)
    log("ETAPA 3 & 4: CONSTRUÇÃO DOS MODELOS COM PIPELINE")
    log("=" * 75)

    preproc_linear, preproc_tree = criar_preprocessadores()

    # 1. Regressão Logística -> ramo linear (pdays discretizado)
    pipe_lr = ImbPipeline([
        ("preproc", preproc_linear),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
    ])
    pipe_lr.fit(X_train, y_train)
    log("-> Regressão Logística (Ramo Linear: pdays Discretizado via KBinsDiscretizer)")

    # 2. Random Forest -> ramo árvore (pdays contínuo)
    pipe_rf = ImbPipeline([
        ("preproc", preproc_tree),
        ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_state)),
    ])
    pipe_rf.fit(X_train, y_train)
    log("-> Random Forest (Ramo Árvore: pdays Contínuo com Split Nativo)")

    # 3. XGBoost + scale_pos_weight, ajustado por GridSearchCV
    ratio_desbalanceamento = (y_train == 0).sum() / (y_train == 1).sum()
    log(f"   [XGBoost] scale_pos_weight calculado automaticamente: {ratio_desbalanceamento:.2f}")

    pipe_xgb = ImbPipeline([
        ("preproc", preproc_tree),
        ("clf", XGBClassifier(scale_pos_weight=ratio_desbalanceamento, eval_metric="logloss",
                              random_state=random_state)),
    ])

    grid_xgb = GridSearchCV(pipe_xgb, config.PARAM_GRID_XGB, cv=config.CV_FOLDS,
                            scoring=config.CV_SCORING, n_jobs=-1)
    grid_xgb.fit(X_train, y_train)
    log(f"-> XGBoost (scale_pos_weight={ratio_desbalanceamento:.2f}) "
        f"(Ramo Árvore Otimizado | CV ROC-AUC: {grid_xgb.best_score_:.4f})")
    log("=" * 75 + "\n")

    return {
        "Regressão Logística": pipe_lr,
        "Random Forest": pipe_rf,
        "XGBoost": grid_xgb.best_estimator_,
    }
