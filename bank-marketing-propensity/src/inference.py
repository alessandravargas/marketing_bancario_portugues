"""Serviço de inferência (porte fiel de `InferenceService`) + persistência."""
from __future__ import annotations

import pandas as pd

from . import config


class InferenceService:
    """Encapsula o pipeline treinado para servir previsões (API ou batch)."""

    def __init__(self, trained_pipeline, limiar: float = config.LIMIAR_DECISAO):
        self.pipeline = trained_pipeline
        self.limiar = limiar

    def prever_propensao_cliente(self, dados_novos_dict: dict) -> dict:
        """Recebe um dicionário com os dados de um cliente; devolve probabilidade e recomendação."""
        df_cliente = pd.DataFrame([dados_novos_dict])
        probabilidade = float(self.pipeline.predict_proba(df_cliente)[:, 1][0])
        decisao = ("Contatar Cliente (Alta Propensão)" if probabilidade >= self.limiar
                   else "Não Contatar (Baixa Propensão)")
        return {
            "probabilidade_conversao": round(probabilidade, 4),
            "recomendacao_acao": decisao,
        }

    def prever_lote(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extensão batch (não existe no notebook): escora uma tabela de clientes."""
        prob = self.pipeline.predict_proba(df)[:, 1]
        out = df.copy()
        out["probabilidade_conversao"] = prob
        out["contatar"] = prob >= self.limiar
        return out


def salvar_modelo(pipeline, caminho):
    import joblib
    joblib.dump(pipeline, caminho)


def carregar_modelo(caminho):
    import joblib
    return joblib.load(caminho)
