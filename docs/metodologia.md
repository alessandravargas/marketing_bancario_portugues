# Metodologia — mapeamento com o roteiro do teste

Cada item do enunciado ("Roteiro do Teste") e onde ele é respondido neste repositório.

| Item do roteiro | Onde está | Resumo |
|---|---|---|
| Carregar dados, estatísticas básicas e tipos | `src/data.py`, `src/eda.py::resumo_basico` | 45.211 linhas × 17 colunas (7 numéricas `int64`, 10 `object`). |
| Valores ausentes | `df.info()` | Nenhum nulo. `unknown` em categóricas é mantido como categoria. `pdays = -1` é código para "nunca contatado" e tem tratamento específico. |
| Visualizar numéricas e categóricas | `src/eda.py` (`plot_target`, `plot_categoricas`, `plot_pdays*`) | Figuras em `reports/figures/`. |
| Relação das variáveis com `y` | `src/eda.py::plot_associacao` | Correlação de Pearson (numéricas) e Informação Mútua (todas). |
| Codificar / normalizar / padronizar | `src/preprocessing.py` | `StandardScaler`, `OneHotEncoder`, `KBinsDiscretizer` (somente `pdays`, ramo linear). |
| Treino / teste | `src/data.py::dividir_treino_teste` | 80/20 estratificado, `random_state=42`. |
| Modelo e otimização | `src/models.py` | LR, RF, XGBoost; grade de 8 combinações × 3 folds para o XGBoost. |
| Métricas e justificativa | `src/evaluation.py` | ROC-AUC, PR-AUC, F1, Recall, Precisão, custo R$. |
| Gráficos explicativos | `src/evaluation.py::plotar_avaliacao_comparativa` | Matrizes de confusão com contagem, % e custo por célula. |
| Features mais importantes | `src/interpretation.py` | Top 8 do XGBoost. |
| Uso em produção | `src/inference.py` + README §4 | `InferenceService`; recomendações de produção sinalizadas como proposta. |

## Por que dois ramos de pré-processamento?

`pdays` vale `-1` para clientes nunca contatados e um número positivo de dias para os demais. Em um modelo linear, `-1` entra como "um dia a menos que zero", o que não tem sentido de negócio. A discretização em 4 faixas uniformes separa esse grupo. Árvores de decisão fazem cortes por limiar e isolam o `-1` sem distorção, então o ramo de árvores mantém a variável contínua.

## Por que PR-AUC e não só ROC-AUC?

Com ≈ 12% de positivos, a ROC-AUC pode parecer alta mesmo com muitos falsos positivos entre os clientes de maior score. A PR-AUC concentra-se na classe rara (precisão × recall), e por isso é o critério de ordenação da tabela de resultados.

## Por que custo-sensível e não SMOTE?

O notebook importa `SMOTE`, mas o balanceamento é feito por pesos (`class_weight='balanced'`; `scale_pos_weight`). Isso evita criar clientes sintéticos e mantém o pipeline simples. Não foi feita comparação experimental com SMOTE.

## Decisões do notebook preservadas sem alteração

- `default` não entra no modelo.
- Limiar 0,5 no `InferenceService`.
- XGBoost fixado como modelo final.
- Custos FP = R$ 15, FN = R$ 150.

## Mudanças deste repositório em relação ao notebook

Apenas organizacionais: código separado em módulos; imports pesados (`xgboost`, `imblearn`) carregados no momento do treino; figuras salvas em disco; função `prever_lote` e `salvar_modelo` adicionadas (extensões, não alteram a modelagem).
