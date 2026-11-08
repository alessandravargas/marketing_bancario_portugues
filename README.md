# Propensão à Assinatura de Depósito a Prazo — Marketing Bancário Português

> Modelo de classificação para priorizar clientes em campanhas de telemarketing de um banco português.
> **Dados:** 45.211 contatos · 16 variáveis · alvo `y` (assinou depósito a prazo: sim/não)

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

---

## 1. Sumário executivo

| | |
|---|---|
| **Problema** | Prever quais clientes têm maior probabilidade de assinar um depósito a prazo, para concentrar o esforço de contato. |
| **Desafio de dados** | Classe positiva rara (≈ 11,7% — razão negativos/positivos de 7,55 no treino). |
| **Abordagem** | Três modelos comparados (Regressão Logística, Random Forest, XGBoost), com balanceamento *cost-sensitive* e dois ramos de pré-processamento. |
| **Modelo final** | **XGBoost** ajustado por `GridSearchCV` (CV ROC-AUC = **0,9319**). |
| **Resultado no teste** | ROC-AUC **0,9305** · PR-AUC **0,6260** · Recall **0,88** · Precisão **0,43** · F1 **0,58**. |
| **Menor custo de negócio** | XGBoost: **R$ 37.290** (vs. R$ 46.650 da Regressão Logística e R$ 107.310 do Random Forest), *sob as premissas de custo abaixo*. |

> A variável `duration` está incluída no modelo e aparece como a 3ª variável mais importante; os resultados apresentados refletem essa inclusão.

Todos os números desta página vêm da execução do notebook original (`notebooks/Bank_Marketing_Data_Analysis.ipynb`); nenhum foi estimado ou inventado.

---

## 2. Resultados

### 2.1 Comparação dos modelos (conjunto de teste, 20%, estratificado)

| Modelo | ROC-AUC | PR-AUC | F1 | Recall | Precisão | Custo de negócio (R$) |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost** | **0,9305** | **0,6260** | **0,5810** | **0,8800** | 0,4336 | **37.290** |
| Random Forest | 0,9271 | 0,6172 | 0,4561 | 0,3384 | **0,6992** | 107.310 |
| Regressão Logística | 0,9094 | 0,5358 | 0,5549 | 0,8195 | 0,4194 | 46.650 |

*Fonte: saída do notebook, etapa 5. Limiar de decisão padrão (0,5).*

**Leitura:**
- O XGBoost lidera em ROC-AUC, PR-AUC, F1 e custo. A diferença de ROC-AUC para o Random Forest é pequena (0,9305 vs. 0,9271).
- O Random Forest tem a maior precisão, mas recall de apenas 34%: deixa passar a maioria dos compradores, o que o torna o mais caro sob a premissa FN = R$ 150.
- XGBoost e Regressão Logística têm perfil parecido (recall alto, precisão ≈ 42–43%): contatam mais clientes para capturar mais compradores.

### 2.2 Variáveis mais importantes (XGBoost)

| # | Variável (após one-hot) | Importância |
|:-:|---|---:|
| 1 | `poutcome_success` | 0,1437 |
| 2 | `contact_unknown` | 0,1260 |
| 3 | `duration` | 0,0706 |
| 4 | `poutcome_unknown` | 0,0645 |
| 5 | `housing_no` | 0,0517 |
| 6 | `month_mar` | 0,0488 |
| 7 | `month_jun` | 0,0410 |
| 8 | `month_oct` | 0,0378 |

*Fonte: saída do notebook, etapa 6. Importância = `feature_importances_` do XGBoost.*

**O que isso sugere (hipóteses a validar, não conclusões causais):**
- **Histórico positivo importa:** ter sucesso na campanha anterior (`poutcome_success`) é o sinal mais forte.
- **Canal/registro de contato:** `contact_unknown` aparece em 2º; o significado de negócio desse sinal não está documentado no dataset e deve ser investigado com o time dono dos dados.
- **Sazonalidade:** meses específicos (mar, jun, out) pesam no modelo.
- **Duração da chamada** aparece entre as 3 principais variáveis em importância.

### 2.3 Sinais da EDA

| Variável | Correlação de Pearson com `y` | Informação Mútua |
|---|---:|---:|
| `duration` | 0,3945 | 0,0705 |
| `pdays` | 0,1036 | 0,0260 |
| `previous` | 0,0932 | 0,0127 |
| `balance` | 0,0528 | 0,0223 |
| `campaign` | −0,0732 | — |
| `poutcome` | n/a (categórica) | 0,0367 |

*Fonte: notebook, célula de EDA (top-10 de MI e correlações das numéricas).*

---

## 3. Metodologia

```
Dados (Kaggle) → EDA → Pré-processamento (2 ramos) → Treino/Tuning → Avaliação (métricas + custo) → Interpretação → Inferência
```

| Etapa | Decisão | Justificativa |
|---|---|---|
| **Valores ausentes** | Nenhum nulo (`df.info()`: 45.211 não-nulos em todas as colunas). Categorias `unknown` mantidas como nível próprio. | Evita descartar linhas; `unknown` pode carregar informação. |
| **Split** | 80/20 estratificado, `random_state=42`. | Preserva a proporção da classe rara. |
| **Numéricas** | `StandardScaler` em `age, balance, day, duration, campaign, previous`. | Necessário para a Regressão Logística; inofensivo para árvores. |
| **Categóricas** | `OneHotEncoder(handle_unknown='ignore')`. | Evita ordem artificial; tolera categorias novas em produção. |
| **`pdays` (ramo linear)** | `KBinsDiscretizer(n_bins=4, uniform, one-hot)`. | `-1` significa "nunca contatado"; tratá-lo como número distorce um modelo linear. |
| **`pdays` (ramo árvore)** | Contínuo + `StandardScaler`. | Árvores separam o valor `-1` nativamente. |
| **`default`** | Descartada pelo `ColumnTransformer` (não listada). | Comportamento do notebook. |
| **Desbalanceamento** | `class_weight='balanced'` (LR, RF) e `scale_pos_weight` = 7,55 (XGBoost). | Aprendizado sensível a custo, sem sintetizar dados. **SMOTE é importado no notebook mas não é usado.** |
| **Tuning** | `GridSearchCV` só para o XGBoost: `n_estimators` {50,100} × `max_depth` {3,5} × `learning_rate` {0,05; 0,1}; `cv=3`, `scoring='roc_auc'`. | LR e RF usam configuração fixa (`max_iter=1000`; `n_estimators=100`). |
| **Métricas** | ROC-AUC, PR-AUC, F1, Recall, Precisão + custo de negócio. | Com ≈ 12% de positivos, acurácia engana; PR-AUC e Recall refletem melhor a classe de interesse. |
| **Custo de negócio** | `FP × R$ 15 + FN × R$ 150`. | Contato desnecessário (tempo de atendente) vs. venda perdida. **Valores são premissas do notebook, não dados reais do banco.** |

Detalhes e justificativas ampliadas: [`docs/metodologia.md`](docs/metodologia.md).

---

## 4. Uso em produção

A classe `InferenceService` (`src/inference.py`) encapsula o pipeline treinado (pré-processamento + modelo) e recebe um cliente como dicionário:

```python
from src.inference import InferenceService
servico = InferenceService(modelo_treinado)
servico.prever_propensao_cliente({"age": 20, "balance": 500.0, "day": 15, "duration": 100,
    "campaign": 1, "pdays": 120, "previous": 2, "job": "management", "marital": "married",
    "education": "tertiary", "housing": "yes", "loan": "yes", "contact": "cellular",
    "month": "mar", "poutcome": "success"})
# {'probabilidade_conversao': 0.6156, 'recomendacao_acao': 'Contatar Cliente (Alta Propensão)'}
```

*O resultado acima (61,56%) é o exemplo do notebook.*

**Caminho recomendado para produção** (proposta de arquitetura; itens marcados com ◦ **não foram implementados** no notebook):

1. **Scoring em lote** — escorar a base antes de cada campanha e ordenar por probabilidade (`prever_lote`, extensão deste repositório).
2. ◦ **Calibrar o limiar** pelo custo (FP = R$ 15 / FN = R$ 150, ou valores reais) em vez do 0,5 fixo.
3. ◦ **Monitorar** deriva de dados e de desempenho (PSI, PR-AUC mensal) e definir gatilho de retreino.
4. ◦ **Teste A/B / grupo controle** para medir o ganho incremental real da campanha.
5. ◦ **Governança**: versionar o pipeline (`joblib`), registrar parâmetros e dados de treino.

---

## 5. Como executar

```bash
git clone https://github.com/alessandravargas/marketing_bancario_portugues.git
cd marketing_bancario_portugues
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Opção A – baixa o dataset automaticamente via kagglehub (pode exigir credenciais Kaggle)
python -m src.main

# Opção B – CSV local (bank-full.csv, separador ';')
python -m src.main --csv data/raw/bank-full.csv

pytest -q          # testes
```

Saídas em `reports/` (`metricas_teste.csv`, `top_features_xgboost.csv`, `exemplo_inferencia.json`) e `reports/figures/` (gráficos de EDA e matrizes de confusão).

> O dataset **não está versionado** neste repositório. Baixe-o na fonte (seção 8).

---

## 6. Estrutura do repositório

```
marketing_bancario_portugues/
├── README.md
├── requirements.txt
├── LICENSE
├── notebooks/Bank_Marketing_Data_Analysis.ipynb   # notebook original (fonte da modelagem)
├── src/
│   ├── config.py          # constantes, colunas, custos, grade do XGBoost
│   ├── data.py            # carga (kagglehub / CSV), split
│   ├── eda.py             # análise exploratória
│   ├── preprocessing.py   # ColumnTransformers (ramo linear e ramo árvore)
│   ├── models.py          # LR, RF, XGBoost + GridSearchCV
│   ├── evaluation.py      # métricas, custo, matrizes de confusão
│   ├── interpretation.py  # importância de variáveis
│   ├── inference.py       # InferenceService
│   └── main.py            # pipeline ponta a ponta
├── tests/                 # testes unitários (dados sintéticos)
├── docs/                  # metodologia e referências
├── data/raw/              # (vazio) coloque aqui bank-full.csv
└── reports/               # saídas geradas
```

---

## 7. Limitações e pontos de atenção

1. **Custos R$ 15 / R$ 150 são premissas**, não números do banco. O ranking de custo entre modelos depende delas.
2. **Limiar fixo em 0,5** — não otimizado.
3. **Só o XGBoost é ajustado** por grid; LR e RF usam configuração fixa, então a comparação favorece o XGBoost.
4. **Comparação e escolha feitas no mesmo conjunto de teste** (o modelo final foi fixado como XGBoost após a avaliação); não há um terceiro conjunto de validação independente.
5. **Sem validação temporal:** o split é aleatório, embora os dados tenham ordem cronológica (`day`, `month`); o desempenho em períodos futuros pode ser menor.
6. **SMOTE importado e não usado**; `default` descartada; ambos no notebook original.
7. O exemplo de inferência passa `default` no dicionário, mas ela é ignorada pelo pipeline.
8. **Não há calibração de probabilidades**: `scale_pos_weight`/`class_weight` deslocam as probabilidades, então o valor "61,56%" deve ser lido como *score*, não como probabilidade real.

---

## 8. Fontes e referências

- **Dataset (Kaggle):** abdelazizsami, *Bank Marketing*. https://www.kaggle.com/datasets/abdelazizsami/bank-marketing
- **Dataset original (UCI):** Bank Marketing. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/222/bank+marketing
- **Artigo de origem dos dados:** Moro, S.; Cortez, P.; Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, 62, 22–31. https://doi.org/10.1016/j.dss.2014.03.001
- **Bibliotecas:** scikit-learn (Pedregosa et al., 2011, JMLR 12:2825–2830); XGBoost (Chen & Guestrin, 2016, KDD '16, https://doi.org/10.1145/2939672.2939785); imbalanced-learn (Lemaître et al., 2017, JMLR 18(17):1–5).
- **Notebook e roteiro do teste:** materiais do projeto "Marketing Bancário Português" (`notebooks/` e enunciado).

## 9. Licença

MIT — ver [LICENSE](LICENSE). O dataset segue os termos de sua fonte (Kaggle/UCI).
