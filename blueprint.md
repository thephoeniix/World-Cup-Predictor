# Blueprint — Predictor del Mundial: ¿las variables macroeconómicas ayudan a predecir fútbol?

Proyecto de investigación comparativo. Construyes **dos modelos sobre el mismo motor Poisson** y mides cuál predice mejor, para responder una pregunta concreta y falsable.

- **Modelo A (Klement+):** Elo + FIFA + PIB per cápita + población + clima
- **Modelo B (Football Analytics):** Elo + valor de mercado + forma reciente + localía
- **Pregunta:** ¿añadir las variables macro mejora el RPS / Log Loss fuera de muestra, o no aportan nada sobre el Elo?

Se evalúa prediciendo **cada partido** de los Mundiales 2010, 2014, 2018, 2022 y la **fase de grupos 2026**, con backtest temporal sin fuga de datos.

---

## 1. Herramientas (stack)

| Capa | Herramienta | Para qué |
|---|---|---|
| Lenguaje | **Python 3.11** | Todo el proyecto |
| Datos | **pandas**, **numpy** | Manipulación; (polars opcional si crece) |
| Descarga | **kagglehub**, **wbdata** / **pandas_datareader** | Bajar datasets Kaggle y la API del Banco Mundial |
| Modelo base | **statsmodels** (GLM Poisson), **scipy.stats** (Poisson PMF) | El motor de goles |
| Modelo avanzado | **PyMC** | Poisson jerárquico bayesiano (la pieza "wow" para aprender) |
| Métricas | implementación propia de **RPS** + **scikit-learn** (log_loss, brier) | Evaluación probabilística |
| Tuning / interpretabilidad | **Optuna**, **SHAP** | Ajustar hiperparámetros y explicar qué features pesan |
| Tracking | **MLflow** o **Weights & Biases** | Registrar experimentos (buena práctica MLOps) |
| Visualización | **matplotlib**, **plotly** | Diagramas de calibración, probabilidades |
| Interfaz | **Streamlit** | Dashboard rápido (alternativa seria: FastAPI + React) |
| Entorno | **venv/conda**, **Jupyter**, **Git/GitHub** | Reproducibilidad y control de versiones |

Instalación base:
```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy statsmodels scipy scikit-learn matplotlib plotly \
            streamlit kagglehub wbdata mlflow pymc optuna shap jupyter
```

---

## 2. Datasets (qué bajar y de dónde)

### Núcleo (compartido + Modelo B)

| Dato | Fuente | Qué saca | Cómo lo usas |
|---|---|---|---|
| **Resultados históricos** | Kaggle `martj42/international-football-results-from-1872-to-2017` (cubre 1872–2026) | date, home/away_team, scores, tournament, neutral | Columna vertebral. De aquí calculas Elo y forma. |
| **Elo** | `eloratings.net`, o **calcúlalo tú** desde los resultados | rating previo a cada partido | Recomendado calcularlo tú: controlas la fuga temporal. |
| **Valor de mercado (selecciones)** | Kaggle `davidcariboo/player-scores` (Transfermarkt) | tablas `national_teams`, `player_valuations` | Agregas valor del plantel por selección y fecha. |
| **Forma reciente** | derivado de los resultados | medias móviles de goles / puntos | No requiere fuente externa. |
| **Localía / anfitrión** | flag `neutral` + lista de sedes por edición | host_diff (+1 local anfitrión, −1 visitante anfitrión) | México/USA/Canadá juegan de locales en 2026. |

Proxy alternativo de calidad de plantel si el de Transfermarkt te cuesta: datasets de **SoFIFA / FIFA videojuego** (`stefanoleone992/fifa-23-complete-player-dataset`), con ratings por jugador.

### Modelo A (Klement+)

| Dato | Fuente | Indicador / nota |
|---|---|---|
| **PIB per cápita** | Banco Mundial (`wbdata`) | indicador `NY.GDP.PCAP.CD` |
| **Población** | Banco Mundial `SP.POP.TOTL`, o UN WPP `population.un.org/wpp` | total y/o franja 15–35 |
| **Clima / temperatura** | World Bank Climate Knowledge Portal `climateknowledgeportal.worldbank.org` | temperatura media por país → idoneidad a la sede |
| **Ranking FIFA** | `inside.fifa.com/fifa-world-ranking/men` o mirror en Kaggle | recuerda: menor número = mejor (hay que invertir) |

### Benchmarks (para tener contra qué comparar)

| Dato | Fuente | Uso |
|---|---|---|
| **Baseline 2026 por Elo** | Kaggle `sarazahran1/wc2026-match-probability-baseline-dataset` | piso que tu modelo debe superar |
| **Cuotas de casas** | `football-data.co.uk` | referencia "techo": las odds implícitas son de los mejores predictores |

> **El paso que más se subestima:** normalizar nombres de equipos entre datasets ("USA"/"United States", nombres históricos). El dataset de martj42 ya unifica al nombre actual; úsalo como tabla canónica y mapea el resto contra él.

---

## 3. Pipeline final

```
[1] Ingesta            Bajar resultados + tablas externas (Kaggle, Banco Mundial)
        │
[2] Limpieza           Normalizar nombres de equipos, parsear fechas, quitar B-teams
        │
[3] Elo dinámico       Recorrer partidos en orden cronológico → Elo PREVIO por partido
        │
[4] Features           Merge tablas externas + construir DIFFS (local−visitante)
        │              Estandarizar (z-score AJUSTADO SOLO EN TRAIN)
        │
[5] Modelo Poisson     log(λ) = β0 + β_local·localía + Σ βi·z(feature_i)
        │              Motor COMPARTIDO; A y B solo cambian la lista de features
        │
[6] Backtest temporal  Por cada Mundial: train = partidos ANTERIORES; predecir el torneo
        │
[7] Evaluación         RPS · Log Loss · Brier · ECE · curva de calibración
        │              Bootstrap pareado A vs B → ¿la diferencia es significativa?
        │
[8] Monte Carlo        Simular torneo 2026 (12 grupos, 8 mejores 3os, R32) → P(campeón)
        │
[9] Interfaz           Streamlit: probabilidades por partido + avance + calibración
        │
[10] Conclusión        Coeficientes + ΔRPS → ¿las macro aportan? (respuesta de la tesis)
```

### Decisiones técnicas que NO debes saltarte

- **Sin fuga temporal.** Para evaluar el Mundial 2018 entrenas SOLO con partidos anteriores a junio 2018. El Elo se construye cronológico. El z-score se ajusta en train y se aplica a test. Esto es lo que separa un resultado real de uno inflado.
- **Mismo motor, distintos features.** A y B comparten el código del Poisson y predicen exactamente los mismos partidos → RPS/LogLoss comparables 1:1.
- **Métrica correcta = RPS**, no accuracy. RPS respeta el orden Local < Empate < Visitante: equivocarte por poco penaliza menos.
- **Dos fuentes de aleatoriedad, separadas.** La del partido la da el Poisson (`Goals ~ Poisson(λ)`). La de tu desconocimiento de la fuerza (epistémica) la metes muestreando los parámetros una vez por torneo simulado (bootstrap del modelo, o posterior si usas PyMC). NO sumes ruido a cada partido: eso duplica varianza y rompe la calibración.
- **Sin DTSI a mano.** Los pesos los aprende el GLM (los coeficientes). Estandariza los features y los coeficientes son importancias comparables → ahí lees si el PIB aporta.
- **Penales solo en eliminatorias** como moneda ligeramente sesgada por Elo. Es el único componente de azar explícito justificado.
- **Significancia.** La diferencia de RPS entre A y B se valida con bootstrap pareado sobre el RPS por-partido. Sin esto confundes ruido con señal.

### Formato 2026 (para el Monte Carlo)

48 equipos → 12 grupos de 4. Avanzan los 2 primeros de cada grupo **+ los 8 mejores terceros** → Ronda de 32 → eliminación directa hasta la final. Cuidado: la selección de los 8 mejores terceros (por puntos, dif. de goles, goles a favor) es la parte que más gente programa mal.

---

## 4. Cómo realizarlo (roadmap por hitos)

**V1 — Núcleo que funciona (empieza aquí)**
Resultados + Elo propio + Poisson GLM + backtest en 2010–2022.
Meta: que tu modelo le gane al baseline de Elo puro y a la tasa base. Si no, hay un bug.

**V2 — Modelo B completo**
Añade valor de mercado + forma reciente + localía. Mide ΔRPS vs V1.

**V3 — Modelo A (Klement+)**
Añade PIB + población + clima + FIFA. Entrena A y B en paralelo sobre los mismos partidos.

**V4 — Comparación rigurosa (el corazón de la tesis)**
RPS/LogLoss/Brier/ECE + curva de calibración + bootstrap pareado por cada Mundial.
Inspecciona coeficientes (¿β del PIB ≈ 0 y no significativo?). Esta es tu respuesta.

**V5 — Monte Carlo 2026**
Simula la fase de grupos 2026 (que se está jugando ahora) y compara probabilidades pre-torneo contra resultados reales conforme ocurren.

**V6 — Interfaz + extras avanzados**
Streamlit con probabilidades y calibración. Opcional para aprender: Poisson jerárquico bayesiano en PyMC (te da la incertidumbre epistémica "gratis"), SHAP para interpretabilidad, MLflow para trackear experimentos.

---

## 5. Estructura de carpetas sugerida

```
wc_predictor/
├── data/
│   ├── raw/            # descargas crudas (Kaggle, Banco Mundial)
│   └── processed/      # tablas limpias y mergeadas
├── src/
│   ├── ingest.py       # descarga y carga
│   ├── clean.py        # normalización de nombres, fechas
│   ├── elo.py          # Elo dinámico cronológico
│   ├── features.py     # construcción de diffs + estandarización
│   ├── poisson_model.py# motor Poisson compartido
│   ├── bayesian_model.py # (V6) versión PyMC
│   ├── simulate.py     # Monte Carlo del torneo 2026
│   ├── metrics.py      # RPS, log loss, brier, ECE, bootstrap
│   └── backtest.py     # backtest temporal + comparador A vs B
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_backtest_AB.ipynb
│   └── 03_simulacion_2026.ipynb
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## 6. Checklist de errores comunes a evitar

- [ ] ¿El Elo y el z-score usan SOLO datos anteriores a cada Mundial evaluado? (fuga)
- [ ] ¿A y B predicen exactamente los mismos partidos?
- [ ] ¿Estás midiendo RPS y no accuracy?
- [ ] ¿Incluiste un baseline (Elo puro / tasa base) para contextualizar?
- [ ] ¿Validaste la diferencia A vs B con bootstrap pareado y no "a ojo"?
- [ ] ¿Una sola fuente de azar por partido (el Poisson), sin ruido extra encima?
- [ ] ¿Los nombres de equipos están unificados entre todos los datasets?
- [ ] ¿La lógica de los 8 mejores terceros del Monte Carlo está bien?

---

## 7. La conclusión que buscas

Al final tendrás, por cada Mundial, una tabla así:

| Mundial | RPS Modelo A | RPS Modelo B | Baseline | ΔRPS (A−B) | IC95% | ¿Significativo? |
|---|---|---|---|---|---|---|

Si ΔRPS no es significativamente distinto de cero y los coeficientes macro de A son pequeños y no significativos → **la respuesta es que las variables macroeconómicas no aportan poder predictivo sobre el Elo**. Un hallazgo así, bien documentado, es más valioso (y más publicable) que el modelo más complejo posible.
```
