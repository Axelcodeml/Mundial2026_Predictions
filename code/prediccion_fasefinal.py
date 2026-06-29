"""
predicciones_fase_final_mundial.py
===================================
Fase final del Mundial 2026 — desde dieciseisavos hasta la final.
  • Evalúa el modelo en los partidos reales de fase de grupos (test set)
  • Predice todos los cruces de eliminatorias (conocidos de la foto)
  • Monte Carlo arrancando desde dieciseisavos (sin simular grupos)
  • Mejoras: modelo de penaltis, intervalos de confianza, análisis de camino
"""

import os, json, math, joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson
from sklearn.model_selection import (RandomizedSearchCV, TimeSeriesSplit,
                                     cross_val_predict, KFold)
from sklearn.metrics import (classification_report, accuracy_score,
                              log_loss, brier_score_loss, confusion_matrix)
from sklearn.calibration import CalibratedClassifierCV

RUTA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RUTA)
os.makedirs('Predicciones', exist_ok=True)

np.random.seed(42)

# ====================================================================
# PARÁMETROS GLOBALES
# ====================================================================

N_SIMULACIONES = 50_000          # Más sims → intervalos más estables
T_GRUPOS       = 1.0
T_CRUCES       = 1.0
DECAY_RECENCIA = 0.0025
FORZAR_REENTRENAMIENTO = False
N_ITER_REG = 300
N_ITER_1X2 = 1000
tweedie_parameter = 1.2
AGRESIVIDAD_GOLES = .6

# ====================================================================
# CRUCES REALES DE DIECISEISAVOS (desde la imagen del bracket)
# ====================================================================
# El orden importa: los pares consecutivos se enfrentan en octavos
# (1v2, 3v4, …), luego en cuartos (ganador12 v ganador34, …) etc.

CRUCES_R32_REALES = [
    # ─── Lado izquierdo del bracket ───
    ('Alemania',      'Paraguay'),
    ('Francia',       'Suecia'),
    ('Sudáfrica',     'Canadá'),
    ('Países Bajos',  'Marruecos'),
    ('Portugal',      'Croacia'),
    ('España',        'Austria'),
    ('EE. UU.',       'Bosnia-Herzegovina'),
    ('Bélgica',       'Senegal'),
    # ─── Lado derecho del bracket ───
    ('Brasil',        'Japón'),
    ('Costa de Marfil','Noruega'),
    ('México',        'Ecuador'),
    ('Inglaterra',    'RD Congo'),
    ('Argentina',     'Cabo Verde'),
    ('Australia',     'Egipto'),
    ('Suiza',         'Argelia'),
    ('Colombia',      'Ghana'),
]

EQUIPOS_CLASIFICADOS = sorted({eq for cruce in CRUCES_R32_REALES for eq in cruce})

FASES_FECHAS = {
    'Dieciseisavos': '28 jun – 3 jul',  'Octavos': '4 – 7 jul',
    'Cuartos':       '9 – 11 jul',      'Semifinales': '14 – 15 jul',
    '3er Puesto':    '18 jul',           'Final': '19 jul',
}

# ====================================================================
# GRIDS DE HIPERPARÁMETROS (sin cambios respecto al original)
# ====================================================================

PARAM_GRID_REG = {
    'n_estimators':     [100, 200, 300, 500],
    'learning_rate':    [0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
    'max_depth':        [2, 3, 5, 8],
    'subsample':        [0.8, 0.9, 1.0],
    'reg_lambda':       [0.1, 0.5, 1.0, 5.0],
    'gamma':            [0, 0.1],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

PARAM_GRID_1X2 = {
    'n_estimators':     [100, 200, 300, 500, 700, 1000],
    'learning_rate':    [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    'max_depth':        [2, 3, 4, 5],
    'subsample':        [0.7, 0.8, 0.9, 1.0],
    'reg_lambda':       [0.1, 0.5, 1.0, 2.0, 5.0],
    'reg_alpha':        [0, 0.1, 0.5, 1.0],
    'gamma':            [0, 0.05, 0.1, 0.2],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
}

# ====================================================================
# FUNCIONES AUXILIARES
# ====================================================================

def asignar_tier(puntos):
    if puntos >= 1700: return 1
    elif puntos >= 1600: return 2
    elif puntos >= 1500: return 3
    else: return 4

mapa_continentes = {
    'República Checa': 'Europa', 'Bosnia-Herzegovina': 'Europa', 'Suiza': 'Europa',
    'Países Bajos': 'Europa', 'Alemania': 'Europa', 'Escocia': 'Europa',
    'Turquía': 'Europa', 'Suecia': 'Europa', 'España': 'Europa',
    'Bélgica': 'Europa', 'Francia': 'Europa', 'Croacia': 'Europa',
    'Austria': 'Europa', 'Portugal': 'Europa', 'Inglaterra': 'Europa',
    'Noruega': 'Europa',
    'Paraguay': 'Sudamérica', 'Brasil': 'Sudamérica', 'Ecuador': 'Sudamérica',
    'Uruguay': 'Sudamérica', 'Argentina': 'Sudamérica', 'Colombia': 'Sudamérica',
    'México': 'Norteamérica', 'Canadá': 'Norteamérica', 'EE. UU.': 'Norteamérica',
    'Haití': 'Norteamérica', 'Curazao': 'Norteamérica', 'Panamá': 'Norteamérica',
    'Sudáfrica': 'Africa', 'Marruecos': 'Africa', 'Egipto': 'Africa',
    'Túnez': 'Africa', 'Costa de Marfil': 'Africa', 'Cabo Verde': 'Africa',
    'Senegal': 'Africa', 'RD Congo': 'Africa', 'Argelia': 'Africa',
    'Ghana': 'Africa',
    'Corea del Sur': 'Asia', 'Catar': 'Asia', 'Japón': 'Asia',
    'Australia': 'Asia', 'Irán': 'Asia', 'Arabia Saudí': 'Asia',
    'Jordania': 'Asia', 'Irak': 'Asia', 'Uzbekistán': 'Asia',
    'Nueva Zelanda': 'Asia',
}

pesos_continente = {
    'Europa': 1.00, 'Sudamérica': 0.95, 'Norteamérica': 0.75,
    'Africa': 0.6, 'Asia': 0.7, 'Oceanía': 0.5,
}


def tunear_temperatura(probs, y_true, name="General"):
    best_t, best_loss = 1.0, float('inf')
    for t in np.arange(0.1, 3.1, 0.01):
        p_t = probs ** (1 / t)
        p_t = p_t / p_t.sum(axis=1, keepdims=True)
        loss = log_loss(y_true, p_t)
        if loss < best_loss:
            best_loss, best_t = loss, t
    print(f"  Temperatura óptima ({name}): T={best_t:.2f} (Log-loss: {best_loss:.4f})")
    return round(best_t, 2)


# ────────────────────────────────────────────────────────────────
# MEJORA 1: Modelo de penaltis para eliminatorias
# ────────────────────────────────────────────────────────────────
# En eliminatorias un empate → prórroga/penaltis. Calculamos la
# probabilidad de avanzar como:
#   P(avanza A) = P(1) + P(X) · P(A gana penaltis)
#
# P(A gana penaltis) combina DOS señales:
#   1. Ranking (efecto psicológico / experiencia en tandas)
#   2. Fuerza relativa del modelo (P(1) vs P(2)): el equipo que
#      domina los 90 min también tiene ventaja en la prórroga
#      y llega con más confianza a los penaltis.

W_RANKING = 0.5     # Peso del componente de ranking
W_MODELO  = 0.5     # Peso del componente del modelo 1X2

def prob_avanzar_knockout(p1, px, p2, puntos_l, puntos_v):
    """Probabilidad de que el equipo local avance en formato knockout."""
    # Componente 1: Ranking (diferencia normalizada, tope ±10%)
    diff_ranking = np.clip((puntos_l - puntos_v) / 400, -0.10, 0.10)
    p_pen_ranking = 0.50 + diff_ranking

    # Componente 2: Fuerza relativa del modelo
    # Si el modelo da P(1)=0.60 y P(2)=0.20, el local es claramente mejor
    # → debería tener ventaja también en tanda (domina prórroga, más confianza)
    p1_vs_p2 = p1 / max(p1 + p2, 1e-9)          # ∈ [0, 1], 0.5 = igualdad
    diff_modelo = np.clip((p1_vs_p2 - 0.5) * 0.20, -0.10, 0.10)  # tope ±10%
    p_pen_modelo = 0.50 + diff_modelo

    # Combinación ponderada + safety bounds
    p_pen_local = W_RANKING * p_pen_ranking + W_MODELO * p_pen_modelo
    p_pen_local = np.clip(p_pen_local, 0.30, 0.70)

    p_avanza_l = p1 + px * p_pen_local
    p_avanza_v = p2 + px * (1 - p_pen_local)
    # Renormalizar por seguridad
    total = p_avanza_l + p_avanza_v
    return p_avanza_l / total, p_avanza_v / total


# ────────────────────────────────────────────────────────────────
# MEJORA 2: Upset score (índice de sorpresa)
# ────────────────────────────────────────────────────────────────

def upset_score(prob_fav, prob_underdog):
    """Score 0-100: cuán probable es la sorpresa. >60 = partido peligroso."""
    if prob_fav <= prob_underdog:
        return 0.0
    ratio = prob_underdog / max(prob_fav, 1e-9)
    return round(min(ratio * 100, 100), 1)


# ====================================================================
# 1. ENTRENAMIENTO (igual que el original)
# ====================================================================

def entrenar_modelos():
    df = pd.read_csv('./Data/datos_historicos.csv')
    df['Fecha_dt'] = pd.to_datetime(df['Fecha'])
    df.sort_values('Fecha_dt', inplace=True)
    df.dropna(inplace=True)

    fecha_max = df['Fecha_dt'].max()
    df['Dias_Antiguedad'] = (fecha_max - df['Fecha_dt']).dt.days
    df['Peso_Recencia']   = np.exp(-DECAY_RECENCIA * df['Dias_Antiguedad'])

    df['Resultado_1X2_Num'] = df['Resultado_1X2'].map({'1': 0, 'X': 1, '2': 2})
    y = df['Resultado_1X2_Num']

    cols_a_excluir = [
        'Fecha', 'Fecha_dt', 'Dias_Antiguedad', 'Peso_Recencia',
        'Equipo_Local', 'Equipo_Visitante',
        'Resultado_1X2', 'Resultado_1X2_Num',
        'Goles_Local', 'Goles_Visitante',
        'Valor_Mercado_Millones_Eur_Local', 'Valor_Mercado_Millones_Eur_Visitante',
        'Puntos_Local', 'Puntos_Visitante',
    ]
    X = df.drop(columns=cols_a_excluir, errors='ignore')

    split_index    = int(len(df) * 0.85)
    X_train        = X.iloc[:split_index];        X_test       = X.iloc[split_index:]
    y_train_1X2    = y.iloc[:split_index];        y_test_1X2   = y.iloc[split_index:]
    y_train_gl     = df['Goles_Local'].iloc[:split_index]
    y_train_gv     = df['Goles_Visitante'].iloc[:split_index]
    peso_rec_train = df['Peso_Recencia'].iloc[:split_index].values
    peso_rec_test  = df['Peso_Recencia'].iloc[split_index:].values

    tscv = TimeSeriesSplit(n_splits=5)

    print(f"Fase 1: Regresores Tweedie (N_ITER_REG={N_ITER_REG})...")
    pesos_train_L = np.where(y_train_gl >= 3, 1.5, 1.0) * peso_rec_train
    pesos_train_V = np.where(y_train_gv >= 3, 1.5, 1.0) * peso_rec_train

    xgb_reg_L = xgb.XGBRegressor(
        objective='reg:tweedie', tweedie_variance_power=tweedie_parameter, random_state=42,
    )
    xgb_reg_V = xgb.XGBRegressor(
        objective='reg:tweedie', tweedie_variance_power=tweedie_parameter, random_state=42,
    )

    search_L = RandomizedSearchCV(
        xgb_reg_L, PARAM_GRID_REG, cv=tscv, n_iter=N_ITER_REG,
        scoring='neg_mean_poisson_deviance', random_state=42, n_jobs=-1, verbose=1,
    )
    search_V = RandomizedSearchCV(
        xgb_reg_V, PARAM_GRID_REG, cv=tscv, n_iter=N_ITER_REG,
        scoring='neg_mean_poisson_deviance', random_state=42, n_jobs=-1, verbose=1,
    )
    search_L.fit(X_train, y_train_gl, sample_weight=pesos_train_L)
    search_V.fit(X_train, y_train_gv, sample_weight=pesos_train_V)

    mejor_modelo_L = search_L.best_estimator_
    mejor_modelo_V = search_V.best_estimator_
    print("  Params Goles_L:", search_L.best_params_)
    print("  Params Goles_V:", search_V.best_params_)

    print(f"\nFase 2: Clasificador 1X2 (N_ITER_1X2={N_ITER_1X2}) + calibración isotónica...")
    kf_meta = KFold(n_splits=5, shuffle=False)

    pred_gl_train = (
        cross_val_predict(mejor_modelo_L, X_train, y_train_gl, cv=kf_meta,
                          params={'sample_weight': pesos_train_L}) ** 1.5
        * X_train['Peso_Local']
    )
    pred_gv_train = (
        cross_val_predict(mejor_modelo_V, X_train, y_train_gv, cv=kf_meta,
                          params={'sample_weight': pesos_train_V}) ** 1.5
        * X_train['Peso_Visitante']
    )
    pred_gl_test = mejor_modelo_L.predict(X_test) ** 1.5 * X_test['Peso_Local']
    pred_gv_test = mejor_modelo_V.predict(X_test) ** 1.5 * X_test['Peso_Visitante']

    X_train_meta = X_train.copy()
    X_train_meta['Pred_Goles_L'] = pred_gl_train
    X_train_meta['Pred_Goles_V'] = pred_gv_train
    X_test_meta = X_test.copy()
    X_test_meta['Pred_Goles_L'] = pred_gl_test
    X_test_meta['Pred_Goles_V'] = pred_gv_test

    xgb_clf_base = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3, base_score=0.5, random_state=42,
    )
    search_1X2 = RandomizedSearchCV(
        xgb_clf_base, PARAM_GRID_1X2, cv=tscv, n_iter=N_ITER_1X2,
        scoring='neg_log_loss', random_state=42, n_jobs=-1, verbose=1,
    )
    search_1X2.fit(X_train_meta, y_train_1X2, sample_weight=peso_rec_train)
    print("  Params 1X2:", search_1X2.best_params_)

    calibrado_test = CalibratedClassifierCV(
        estimator=search_1X2.best_estimator_, method='isotonic', cv=tscv,
    )
    calibrado_test.fit(X_train_meta, y_train_1X2, sample_weight=peso_rec_train)
    pred_test_clases = calibrado_test.predict(X_test_meta)
    probs_test       = calibrado_test.predict_proba(X_test_meta)

    print("\n--- RESULTADOS EN TEST HISTÓRICO (15 % temporal) ---")
    print(classification_report(y_test_1X2, pred_test_clases))
    print(f"Accuracy test histórico: {accuracy_score(y_test_1X2, pred_test_clases):.3f}")

    print("\nOptimizando Temperatura (T)...")
    t_optimo = tunear_temperatura(probs_test, y_test_1X2, name="Histórico Global")

    print("\nFase 3: Reentrenando con el 100 % de los datos...")
    y_gl_full     = df['Goles_Local']
    y_gv_full     = df['Goles_Visitante']
    peso_rec_full = df['Peso_Recencia'].values
    pesos_full_L  = np.where(y_gl_full >= 3, 1.5, 1.0) * peso_rec_full
    pesos_full_V  = np.where(y_gv_full >= 3, 1.5, 1.0) * peso_rec_full

    kf_meta_full = KFold(n_splits=5, shuffle=False)
    pred_gl_full = cross_val_predict(mejor_modelo_L, X, y_gl_full, cv=kf_meta_full,
                                     params={'sample_weight': pesos_full_L})
    pred_gv_full = cross_val_predict(mejor_modelo_V, X, y_gv_full, cv=kf_meta_full,
                                     params={'sample_weight': pesos_full_V})

    X_meta_full = X.copy()
    X_meta_full['Pred_Goles_L'] = pred_gl_full
    X_meta_full['Pred_Goles_V'] = pred_gv_full

    mejor_modelo_L.fit(X, y_gl_full, sample_weight=pesos_full_L)
    mejor_modelo_V.fit(X, y_gv_full, sample_weight=pesos_full_V)

    modelo_1X2_final = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3, base_score=0.5,
        random_state=42, **search_1X2.best_params_,
    )
    modelo_1X2_final.fit(X_meta_full, y, sample_weight=peso_rec_full)

    clasificador_calibrado_final = CalibratedClassifierCV(
        estimator=modelo_1X2_final, method='isotonic', cv=tscv,
    )
    clasificador_calibrado_final.fit(X_meta_full, y, sample_weight=peso_rec_full)

    # Importancia de variables
    columnas_1X2 = list(X.columns) + ['Pred_Goles_L', 'Pred_Goles_V']
    importancias_1X2 = modelo_1X2_final.feature_importances_
    df_imp = pd.DataFrame({'Variable': columnas_1X2, 'Importancia': importancias_1X2})
    df_imp = df_imp.sort_values('Importancia', ascending=False).head(25)
    print("\nTop 25 variables más importantes (Modelo 1X2 Final):")
    print(df_imp.to_string(index=False))
    df_imp.to_csv('Predicciones/importancia_variables_top25.csv', index=False, encoding='utf-8-sig')

    joblib.dump(mejor_modelo_L,              'modelo_goles_L.pkl')
    joblib.dump(mejor_modelo_V,              'modelo_goles_V.pkl')
    joblib.dump(clasificador_calibrado_final,'modelo_1X2_calibrado.pkl')
    joblib.dump(list(X.columns),             'columnas_entrenamiento.pkl')

    mejores_params = {
        'N_ITER_REG': N_ITER_REG, 'N_ITER_1X2': N_ITER_1X2,
        'DECAY_RECENCIA': DECAY_RECENCIA, 'T_OPTIMA_CALCULADA': t_optimo,
        'Goles_L': search_L.best_params_, 'Goles_V': search_V.best_params_,
        '1X2': search_1X2.best_params_,
        'score_GL': float(search_L.best_score_),
        'score_GV': float(search_V.best_score_),
        'score_1X2': float(search_1X2.best_score_),
    }
    with open('Predicciones/mejores_hiperparametros.json', 'w', encoding='utf-8') as f:
        json.dump(mejores_params, f, ensure_ascii=False, indent=2)

    print("\nModelos guardados.")


# ====================================================================
# 2. PIPELINE DE PREDICCIÓN (sin cambios)
# ====================================================================

_CACHE_MODELOS = {}

def _modelos():
    if not _CACHE_MODELOS:
        _CACHE_MODELOS['L']    = joblib.load('modelo_goles_L.pkl')
        _CACHE_MODELOS['V']    = joblib.load('modelo_goles_V.pkl')
        _CACHE_MODELOS['1X2']  = joblib.load('modelo_1X2_calibrado.pkl')
        _CACHE_MODELOS['cols'] = joblib.load('columnas_entrenamiento.pkl')
        if os.path.exists('Predicciones/mejores_hiperparametros.json'):
            with open('Predicciones/mejores_hiperparametros.json', encoding='utf-8') as f:
                _CACHE_MODELOS['t_optima'] = json.load(f).get('T_OPTIMA_CALCULADA', 1.0)
        else:
            _CACHE_MODELOS['t_optima'] = 1.0
    return _CACHE_MODELOS


def pipeline_prediccion(df_bruto, sede_neutral=True, T=None):
    m = _modelos()
    modelo_L, modelo_V, modelo_1X2, columnas_base = m['L'], m['V'], m['1X2'], m['cols']
    if T is None:
        T = m.get('t_optima', 1.0)

    def obtener_predicciones_crudas(df_temp):
        df_calc = df_temp.copy()

        cols_avg_local = [
            c for c in df_calc.columns
            if c.endswith(('_5_Local', '_3_Local', '_2_Local', '_total_Local'))
            and c.startswith('avg_')
        ]
        for col_local in cols_avg_local:
            col_vis = col_local.replace('_Local', '_Visitante')
            if col_vis in df_calc.columns:
                nombre = col_local.replace('_Local', '')
                df_calc[f'diff_{nombre}'] = df_calc[col_local] - df_calc[col_vis]

        for sufijo in ['trend_xG_5', 'avg_xGA_2', 'avg_xGA_5', 'avg_xGA_total',
                        'clean_sheet_rate_5', 'forma_vs_historia']:
            col_l = f'{sufijo}_Local'
            col_v = f'{sufijo}_Visitante'
            if col_l in df_calc.columns and col_v in df_calc.columns:
                df_calc[f'diff_{sufijo}'] = df_calc[col_l] - df_calc[col_v]

        col_pts_l = 'avg_Puntos_total_Local'
        col_pts_v = 'avg_Puntos_total_Visitante'
        if col_pts_l in df_calc.columns and col_pts_v in df_calc.columns:
            df_calc['diff_Puntos']        = df_calc[col_pts_l] - df_calc[col_pts_v]
            df_calc['Prob_Implicita_ELO'] = 1 / (1 + 10 ** (-df_calc['diff_Puntos'] / 400))
            df_calc['diff_Tier']          = (
                df_calc[col_pts_l].apply(asignar_tier)
                - df_calc[col_pts_v].apply(asignar_tier)
            )

        if ('Valor_Mercado_Millones_Eur_Local'    in df_calc.columns and
                'Valor_Mercado_Millones_Eur_Visitante' in df_calc.columns):
            df_calc['diff_Valor_Mercado'] = (
                df_calc['Valor_Mercado_Millones_Eur_Local']
                - df_calc['Valor_Mercado_Millones_Eur_Visitante']
            )

        df_calc['Continente_Local']     = df_calc['Equipo_Local'].map(mapa_continentes)
        df_calc['Continente_Visitante'] = df_calc['Equipo_Visitante'].map(mapa_continentes)
        df_calc['Peso_Local']           = df_calc['Continente_Local'].map(pesos_continente)
        df_calc['Peso_Visitante']       = df_calc['Continente_Visitante'].map(pesos_continente)
        df_calc.drop(['Continente_Local', 'Continente_Visitante'], axis=1, inplace=True)

        for col in columnas_base:
            if col not in df_calc.columns:
                df_calc[col] = 0
        X_listo = df_calc[columnas_base]

        goles_L = modelo_L.predict(X_listo)
        goles_V = modelo_V.predict(X_listo)

        X_meta = X_listo.copy()
        X_meta['Pred_Goles_L'] = goles_L
        X_meta['Pred_Goles_V'] = goles_V
        probs = modelo_1X2.predict_proba(X_meta)
        return goles_L, goles_V, probs

    df_normal    = df_bruto.copy()
    cols_contexto = ['Fecha', 'Equipo_Local', 'Equipo_Visitante']
    if 'Grupo' in df_normal.columns:
        cols_contexto.append('Grupo')
    contexto = df_normal[cols_contexto].copy()

    goles_L_norm, goles_V_norm, probs_norm = obtener_predicciones_crudas(df_normal)

    if not sede_neutral:
        resultados = contexto.copy()
        resultados['xG_Modelo_Local']     = goles_L_norm.round(2)
        resultados['xG_Modelo_Visitante'] = goles_V_norm.round(2)
        resultados['Prob_Local']          = probs_norm[:, 0]
        resultados['Prob_Empate']         = probs_norm[:, 1]
        resultados['Prob_Visitante']      = probs_norm[:, 2]
        return resultados

    df_inverso      = df_bruto.copy()
    nuevas_columnas = []
    for col in df_inverso.columns:
        if col.endswith('_Local'):      nuevas_columnas.append(col.replace('_Local',     '_Visitante'))
        elif col.endswith('_Visitante'): nuevas_columnas.append(col.replace('_Visitante', '_Local'))
        else:                           nuevas_columnas.append(col)
    df_inverso.columns = nuevas_columnas
    goles_L_inv, goles_V_inv, probs_inv = obtener_predicciones_crudas(df_inverso)

    resultados = contexto.copy()
    resultados['xG_Modelo_Local']     = ((goles_L_norm + goles_V_inv) / 2).round(2)
    resultados['xG_Modelo_Visitante'] = ((goles_V_norm + goles_L_inv) / 2).round(2)
    resultados['Prob_Local']          = (probs_norm[:, 0] + probs_inv[:, 2]) / 2
    resultados['Prob_Empate']         = (probs_norm[:, 1] + probs_inv[:, 1]) / 2
    resultados['Prob_Visitante']      = (probs_norm[:, 2] + probs_inv[:, 0]) / 2

    probs_afiladas = resultados[['Prob_Local', 'Prob_Empate', 'Prob_Visitante']] ** (1 / T)
    s = probs_afiladas.sum(axis=1).replace(0, 1e-12)
    resultados[['Prob_Local', 'Prob_Empate', 'Prob_Visitante']] = probs_afiladas.div(s, axis=0)
    return resultados


# ====================================================================
# 3. CARGA DE DATOS PARA FASE FINAL
# ====================================================================

def cargar_datos_mundial():
    """Carga datos_mundial.csv y devuelve df_vars limpio + puntos por equipo."""
    df_vars = pd.read_csv('./Data/datos_mundial.csv').sort_values('Fecha')
    df_vars = df_vars.drop_duplicates('Equipo', keep='last').reset_index(drop=True)
    cols_numericas = [c for c in df_vars.columns if c not in ['Equipo', 'Fecha', 'Grupo', 'Continente']]
    for col in cols_numericas:
        df_vars[col] = pd.to_numeric(df_vars[col], errors='coerce').fillna(0)

    # Extraer puntos por equipo para modelo de penaltis
    puntos = {}
    pts_col = [c for c in df_vars.columns if 'Puntos' in c and 'avg' in c and 'total' in c]
    if pts_col:
        for _, r in df_vars.iterrows():
            puntos[r['Equipo']] = r[pts_col[0]]
    return df_vars, puntos


# ────────────────────────────────────────────────────────────────
# H2H: carga y lookup del historial directo entre equipos
# ────────────────────────────────────────────────────────────────

_H2H_CACHE = None

def cargar_h2h():
    """Carga el lookup H2H desde h2h_mundial.csv (generado por Data_Cleaning)."""
    global _H2H_CACHE
    if _H2H_CACHE is not None:
        return _H2H_CACHE
    try:
        df_h2h = pd.read_csv('./Data/h2h_mundial.csv')
        _H2H_CACHE = {(r['Equipo_A'], r['Equipo_B']): r for _, r in df_h2h.iterrows()}
        print(f"  → H2H cargado: {len(_H2H_CACHE)} pares")
    except FileNotFoundError:
        print("  ⚠ h2h_mundial.csv no encontrado — H2H features serán neutras")
        _H2H_CACHE = {}
    return _H2H_CACHE


def obtener_h2h(eq_l, eq_v, h2h_lookup):
    """Devuelve stats H2H desde la perspectiva del equipo local."""
    pair = tuple(sorted([eq_l, eq_v]))
    if pair not in h2h_lookup:
        return {'h2h_win_rate_Local': 1/3, 'h2h_draw_rate': 1/3,
                'h2h_win_rate_Visitante': 1/3,
                'h2h_goles_avg_Local': 1.0, 'h2h_goles_avg_Visitante': 1.0,
                'h2h_n_matches': 0}
    row = h2h_lookup[pair]
    if eq_l == pair[0]:
        return {'h2h_win_rate_Local': row['wins_A'], 'h2h_draw_rate': row['draws'],
                'h2h_win_rate_Visitante': row['wins_B'],
                'h2h_goles_avg_Local': row['goles_avg_A'],
                'h2h_goles_avg_Visitante': row['goles_avg_B'],
                'h2h_n_matches': row['n_matches']}
    else:
        return {'h2h_win_rate_Local': row['wins_B'], 'h2h_draw_rate': row['draws'],
                'h2h_win_rate_Visitante': row['wins_A'],
                'h2h_goles_avg_Local': row['goles_avg_B'],
                'h2h_goles_avg_Visitante': row['goles_avg_A'],
                'h2h_n_matches': row['n_matches']}


def construir_enfrentamientos(cruces, df_vars):
    """Construye DataFrame de features para una lista de cruces [(local, visitante), ...]."""
    df_vars_clean = df_vars.drop(columns=['Fecha'], errors='ignore')
    partido = pd.DataFrame({
        'Fecha':            ['2026-07-01'] * len(cruces),
        'Equipo_Local':     [a for a, b in cruces],
        'Equipo_Visitante': [b for a, b in cruces],
    })
    partido_vars = partido.merge(
        df_vars_clean, left_on='Equipo_Local', right_on='Equipo', how='left'
    ).drop(columns=['Equipo'])
    partido_vars = partido_vars.merge(
        df_vars_clean, left_on='Equipo_Visitante', right_on='Equipo', how='left',
        suffixes=('_Local', '_Visitante')
    ).drop(columns=['Equipo'])

    # ── Añadir H2H ──
    h2h_lookup = cargar_h2h()
    h2h_rows = [obtener_h2h(a, b, h2h_lookup) for a, b in cruces]
    df_h2h = pd.DataFrame(h2h_rows)
    for col in df_h2h.columns:
        partido_vars[col] = df_h2h[col].values
    partido_vars['diff_h2h_win_rate'] = (
        partido_vars['h2h_win_rate_Local'] - partido_vars['h2h_win_rate_Visitante']
    )
    partido_vars['diff_h2h_goles'] = (
        partido_vars['h2h_goles_avg_Local'] - partido_vars['h2h_goles_avg_Visitante']
    )

    return partido_vars


# ====================================================================
# 4. EVALUACIÓN EN FASE DE GRUPOS (MEJORA: backtest real)
# ====================================================================

def evaluar_fase_grupos(T):
    """Evalúa el modelo en los partidos reales de la fase de grupos del Mundial.
    Devuelve la T óptima calibrada sobre estos partidos."""
    print("\n" + "="*70)
    print("  EVALUACIÓN DEL MODELO EN FASE DE GRUPOS DEL MUNDIAL (TEST SET)")
    print("="*70)

    df_pm = pd.read_csv('./Data/partidos_mundial.csv')
    df_pm['Jugado'] = df_pm['Jugado'].astype(str).str.lower() == 'true'
    df_jugados = df_pm[df_pm['Jugado']].copy()

    if df_jugados.empty:
        print("  No hay partidos jugados para evaluar.")
        return T

    # Convertir resultado real a numérico
    df_jugados['Resultado_1X2_Num'] = df_jugados['Resultado_1X2'].map({'1': 0, 'X': 1, '2': 2})
    df_jugados = df_jugados.dropna(subset=['Resultado_1X2_Num'])
    y_real = df_jugados['Resultado_1X2_Num'].astype(int).values
    goles_l_real = df_jugados['Goles_Local'].values.astype(float)
    goles_v_real = df_jugados['Goles_Visitante'].values.astype(float)

    # Construir features y predecir con T original
    df_vars, _ = cargar_datos_mundial()
    cruces = list(zip(df_jugados['Equipo_Local'], df_jugados['Equipo_Visitante']))
    df_feat = construir_enfrentamientos(cruces, df_vars)
    df_pred = pipeline_prediccion(df_feat, sede_neutral=True, T=T)

    probs = df_pred[['Prob_Local', 'Prob_Empate', 'Prob_Visitante']].values
    pred_clases = probs.argmax(axis=1)

    # ── Métricas 1X2 con T original ──
    print(f"\n  Partidos evaluados: {len(y_real)}")
    print(f"\n  {'─'*50}")
    print(f"  CLASIFICACIÓN 1X2 (T original = {T:.2f})")
    print(f"  {'─'*50}")
    target_names = ['1 (Local)', 'X (Empate)', '2 (Visitante)']
    print(classification_report(y_real, pred_clases, target_names=target_names, zero_division=0))
    print(f"  Accuracy:  {accuracy_score(y_real, pred_clases):.3f}")
    print(f"  Log-Loss:  {log_loss(y_real, probs):.4f}")

    # Brier score multiclase
    brier = 0
    for i in range(3):
        y_bin = (y_real == i).astype(float)
        brier += brier_score_loss(y_bin, probs[:, i])
    print(f"  Brier (multi): {brier / 3:.4f}")

    # ── Métricas de goles ──
    xg_l = df_pred['xG_Modelo_Local'].values.astype(float)
    xg_v = df_pred['xG_Modelo_Visitante'].values.astype(float)
    mae_l = np.mean(np.abs(xg_l - goles_l_real))
    mae_v = np.mean(np.abs(xg_v - goles_v_real))
    print(f"\n  {'─'*50}")
    print(f"  PREDICCIÓN DE GOLES")
    print(f"  {'─'*50}")
    print(f"  MAE Goles Local:     {mae_l:.3f}")
    print(f"  MAE Goles Visitante: {mae_v:.3f}")
    print(f"  Media goles reales:  {(goles_l_real.mean() + goles_v_real.mean()):.2f} por partido")
    print(f"  Media xG predichos:  {(xg_l.mean() + xg_v.mean()):.2f} por partido")

    # ── Encontrar T óptima en datos del Mundial ──
    # Necesitamos las probabilidades CRUDAS (antes de aplicar T) para poder
    # re-escalar con distintas T. Las obtenemos pidiendo T=1.0 al pipeline.
    df_pred_raw = pipeline_prediccion(df_feat, sede_neutral=True, T=1.0)
    probs_raw = df_pred_raw[['Prob_Local', 'Prob_Empate', 'Prob_Visitante']].values

    best_t_wc, best_ll = 1.0, float('inf')
    for t_test in np.arange(0.3, 3.0, 0.01):
        p_t = probs_raw ** (1 / t_test)
        p_t = p_t / p_t.sum(axis=1, keepdims=True)
        ll = log_loss(y_real, p_t)
        if ll < best_ll:
            best_ll, best_t_wc = ll, t_test
    best_t_wc = round(best_t_wc, 2)

    print(f"\n  {'─'*50}")
    print(f"  OPTIMIZACIÓN DE TEMPERATURA")
    print(f"  {'─'*50}")
    print(f"  T del entrenamiento histórico: {T:.2f}  (Log-loss: {log_loss(y_real, probs):.4f})")
    print(f"  T óptima en fase de grupos:    {best_t_wc:.2f}  (Log-loss: {best_ll:.4f})")

    # ── Recalcular probabilidades con T óptima ──
    probs_opt = probs_raw ** (1 / best_t_wc)
    probs_opt = probs_opt / probs_opt.sum(axis=1, keepdims=True)
    pred_clases_opt = probs_opt.argmax(axis=1)

    print(f"\n  {'─'*50}")
    print(f"  CALIBRACIÓN COMPARADA (T={T:.2f} → T={best_t_wc:.2f})")
    print(f"  {'─'*50}")
    max_probs_orig = probs.max(axis=1)
    max_probs_opt  = probs_opt.max(axis=1)
    bins = [(0.33, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.80), (0.80, 1.01)]
    print(f"    {'Banda':<18s} {'n':>3s}  {'Acierto_T_orig':>14s}  {'Acierto_T_opt':>14s}  {'Conf_T_opt':>10s}")
    for lo, hi in bins:
        # Usamos las bandas de T óptima para la comparación
        mask = (max_probs_opt >= lo) & (max_probs_opt < hi)
        if mask.sum() == 0:
            continue
        acierto_orig = (pred_clases[mask] == y_real[mask]).mean()
        acierto_opt  = (pred_clases_opt[mask] == y_real[mask]).mean()
        conf_opt     = max_probs_opt[mask].mean()
        print(f"    [{lo:.0%}-{hi:.0%})       {mask.sum():>3d}  {acierto_orig:>13.1%}  {acierto_opt:>13.1%}  {conf_opt:>9.1%}")

    print(f"\n  Accuracy con T original: {accuracy_score(y_real, pred_clases):.3f}")
    print(f"  Accuracy con T óptima:   {accuracy_score(y_real, pred_clases_opt):.3f}")

    # ── Detalle por partido (con T óptima) ──
    print(f"\n  {'─'*50}")
    print(f"  DETALLE POR PARTIDO (T={best_t_wc:.2f})")
    print(f"  {'─'*50}")
    etiquetas = {0: '1', 1: 'X', 2: '2'}
    ok = 0
    for i, (_, row) in enumerate(df_jugados.iterrows()):
        real = etiquetas[y_real[i]]
        pred = etiquetas[pred_clases_opt[i]]
        marca = '✓' if real == pred else '✗'
        ok += (real == pred)
        print(f"    {marca}  {row['Equipo_Local']:>20s} {int(goles_l_real[i])}-{int(goles_v_real[i])} "
              f"{row['Equipo_Visitante']:<20s}  real={real}  pred={pred}  "
              f"({probs_opt[i,0]:.0%}/{probs_opt[i,1]:.0%}/{probs_opt[i,2]:.0%})")
    print(f"\n  Aciertos: {ok}/{len(y_real)} ({ok/len(y_real):.1%})")

    print(f"\n  ★ T={best_t_wc:.2f} se usará para las predicciones de eliminatorias ★")
    return best_t_wc


# ====================================================================
# 5. MATRIZ DE CRUCES (solo 32 clasificados)
# ====================================================================

def matriz_cruces(equipos, df_vars, T_cruces):
    pares = [(a, b) for a in equipos for b in equipos if a != b]

    # Usa construir_enfrentamientos → incluye H2H automáticamente
    partido_vars = construir_enfrentamientos(pares, df_vars)
    df_pred = pipeline_prediccion(partido_vars, sede_neutral=True, T=T_cruces)

    idx = {e: i for i, e in enumerate(equipos)}
    n   = len(equipos)
    P1  = np.zeros((n, n)); PX = np.zeros((n, n)); P2  = np.zeros((n, n))
    XGL = np.zeros((n, n)); XGV = np.zeros((n, n))

    for (a, b), (_, r) in zip(pares, df_pred.iterrows()):
        i, j   = idx[a], idx[b]
        s      = r['Prob_Local'] + r['Prob_Empate'] + r['Prob_Visitante']
        P1[i, j]  = r['Prob_Local']    / s
        PX[i, j]  = r['Prob_Empate']   / s
        P2[i, j]  = r['Prob_Visitante'] / s
        XGL[i, j] = r['xG_Modelo_Local']
        XGV[i, j] = r['xG_Modelo_Visitante']

    # MEJORA: M_adv usa modelo de penaltis con puntos de ranking
    pts_col = [c for c in df_vars.columns if 'Puntos' in c and 'avg' in c and 'total' in c]
    puntos_vec = np.zeros(n)
    if pts_col:
        for _, r in df_vars.iterrows():
            if r['Equipo'] in idx:
                puntos_vec[idx[r['Equipo']]] = r[pts_col[0]]

    M_adv = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                M_adv[i, j] = 0.5
            else:
                M_adv[i, j], _ = prob_avanzar_knockout(
                    P1[i, j], PX[i, j], P2[i, j], puntos_vec[i], puntos_vec[j]
                )

    return {'P1': P1, 'PX': PX, 'P2': P2, 'XGL': XGL, 'XGV': XGV, 'M_adv': M_adv, 'idx': idx}


# ====================================================================
# 6. MARCADOR MÁS PROBABLE (sin cambios)
# ====================================================================

def _poisson_pmf(k, lam):
    lam = max(lam, 0.05)
    return math.exp(-lam) * lam ** k / math.factorial(k)


def marcador_mas_probable(xg_l, xg_v, prob_1, prob_x, prob_2, resultado, agresividad=AGRESIVIDAD_GOLES):
    lam_l = xg_l
    lam_v = xg_v
    if prob_1 > prob_2:
        lam_l = max(xg_l * (1 + (prob_1 * agresividad)), 0.1)
        lam_v = max(xg_v * (1 + (prob_2 * agresividad)/4), 0.1)
    if prob_2 > prob_1:
        lam_l = max(xg_l * (1 + (prob_1 * agresividad/4)), 0.1)
        lam_v = max(xg_v * (1 + (prob_2 * agresividad)), 0.1)

    if resultado == 'X':
        empuje_empate = prob_x * 0.6
        lam_l += empuje_empate
        lam_v += empuje_empate

    mejor, p_mejor = (0, 0), -1.0
    for i in range(9):
        for j in range(9):
            if resultado == '1' and not i > j: continue
            if resultado == 'X' and i != j:   continue
            if resultado == '2' and not i < j: continue
            p = _poisson_pmf(i, lam_l) * _poisson_pmf(j, lam_v)
            if p > p_mejor:
                mejor, p_mejor = (i, j), p
    return mejor


# ====================================================================
# 7. PREDICCIÓN PUNTUAL DE ELIMINATORIAS
# ====================================================================

def prediccion_eliminatorias(mc, puntos):
    """Predice todos los cruces desde R32 hasta la Final con marcadores."""
    idx = mc['idx']
    P1, PX, P2, XGL, XGV = mc['P1'], mc['PX'], mc['P2'], mc['XGL'], mc['XGV']
    registro = []

    def jugar_cruce(fase, eq_a, eq_b):
        i, j       = idx[eq_a], idx[eq_b]
        p1, px, p2 = P1[i, j], PX[i, j], P2[i, j]
        pts_a      = puntos.get(eq_a, 1500)
        pts_b      = puntos.get(eq_b, 1500)

        # Resultado 90 min
        res_90 = ['1', 'X', '2'][int(np.argmax([p1, px, p2]))]
        gl, gv = marcador_mas_probable(XGL[i, j], XGV[i, j], p1, px, p2, res_90)

        # Quién avanza (con modelo de penaltis si empate)
        p_adv_a, p_adv_b = prob_avanzar_knockout(p1, px, p2, pts_a, pts_b)
        if res_90 == 'X':
            ganador  = eq_a if p_adv_a >= p_adv_b else eq_b
            marcador = f'{gl}-{gv} (pen)'
            detalle  = 'Empate → penaltis'
        else:
            ganador  = eq_a if res_90 == '1' else eq_b
            marcador = f'{gl}-{gv}'
            detalle  = 'Tiempo regular'

        # Upset score
        fav_prob = max(p_adv_a, p_adv_b)
        und_prob = min(p_adv_a, p_adv_b)
        u_score  = upset_score(fav_prob, und_prob)

        registro.append({
            'Fase': fase, 'Fechas': FASES_FECHAS.get(fase, ''),
            'Local': eq_a, 'Visitante': eq_b,
            'Marcador': marcador, 'Avanza': ganador,
            'xG_L': round(XGL[i, j], 2), 'xG_V': round(XGV[i, j], 2),
            'Prob_1': round(p1 * 100, 1), 'Prob_X': round(px * 100, 1),
            'Prob_2': round(p2 * 100, 1),
            'P_Avanza_L': round(p_adv_a * 100, 1),
            'P_Avanza_V': round(p_adv_b * 100, 1),
            'Upset': u_score,
            'Detalle': detalle,
        })
        return ganador

    # R32
    g32 = [jugar_cruce('Dieciseisavos', a, b) for a, b in CRUCES_R32_REALES]
    # R16: pares consecutivos
    g16 = [jugar_cruce('Octavos', g32[i], g32[i + 1]) for i in range(0, 16, 2)]
    # QF
    gqf = [jugar_cruce('Cuartos', g16[i], g16[i + 1]) for i in range(0, 8, 2)]
    # SF
    sf = [jugar_cruce('Semifinales', gqf[0], gqf[1]),
          jugar_cruce('Semifinales', gqf[2], gqf[3])]
    # 3er puesto
    perdedores_sf = [e for par in [(gqf[0], gqf[1]), (gqf[2], gqf[3])] for e in par if e not in sf]
    jugar_cruce('3er Puesto', perdedores_sf[0], perdedores_sf[1])
    # Final
    campeon = jugar_cruce('Final', sf[0], sf[1])

    return pd.DataFrame(registro), campeon


# ====================================================================
# 8. MONTE CARLO DESDE DIECISEISAVOS (MEJORA: sin simular grupos)
# ====================================================================

def monte_carlo_fase_final(mc, puntos, n_sims=N_SIMULACIONES):
    """Monte Carlo partiendo del bracket real de R32, sin simular fase de grupos."""
    idx  = mc['idx']
    M    = mc['M_adv']
    n_eq = len(EQUIPOS_CLASIFICADOS)

    contadores = {f: np.zeros(len(idx), dtype=np.int64)
                  for f in ['R32', 'R16', 'QF', 'SF', 'Final', 'Campeon']}

    # Todos los clasificados llegan al R32 con certeza
    for eq in EQUIPOS_CLASIFICADOS:
        contadores['R32'][idx[eq]] = n_sims

    def jugar_n(a_ids, b_ids):
        p = M[a_ids, b_ids]
        return np.where(np.random.random(n_sims) < p, a_ids, b_ids)

    # R32: los cruces son fijos (conocidos)
    r32_ids = [(np.full(n_sims, idx[a], dtype=int),
                np.full(n_sims, idx[b], dtype=int))
               for a, b in CRUCES_R32_REALES]

    g32 = [jugar_n(a, b) for a, b in r32_ids]
    for w in g32:
        np.add.at(contadores['R16'], w, 1)

    # R16
    g16 = [jugar_n(g32[i], g32[i + 1]) for i in range(0, 16, 2)]
    for w in g16:
        np.add.at(contadores['QF'], w, 1)

    # QF
    gqf = [jugar_n(g16[i], g16[i + 1]) for i in range(0, 8, 2)]
    for w in gqf:
        np.add.at(contadores['SF'], w, 1)

    # SF
    sf1 = jugar_n(gqf[0], gqf[1])
    sf2 = jugar_n(gqf[2], gqf[3])
    np.add.at(contadores['Final'], sf1, 1)
    np.add.at(contadores['Final'], sf2, 1)

    # Final
    campeon = jugar_n(sf1, sf2)
    np.add.at(contadores['Campeon'], campeon, 1)

    # Construir tabla solo para clasificados
    inv_idx = {v: k for k, v in idx.items()}
    tabla = pd.DataFrame(
        {f: contadores[f] / n_sims * 100 for f in contadores},
        index=[inv_idx[i] for i in range(len(idx))],
    )
    tabla = tabla[tabla['R32'] > 0]
    return tabla.sort_values(['Campeon', 'Final', 'SF'], ascending=False).round(1)


# ────────────────────────────────────────────────────────────────
# MEJORA 3: Intervalos de confianza por bootstrap
# ────────────────────────────────────────────────────────────────

def bootstrap_confianza(mc, puntos, n_bootstrap=10, n_sims_por_boot=10_000):
    """Ejecuta Monte Carlo múltiples veces para obtener intervalos de confianza."""
    campeon_counts = {}
    for b in range(n_bootstrap):
        np.random.seed(42 + b)
        tabla = monte_carlo_fase_final(mc, puntos, n_sims=n_sims_por_boot)
        for eq in tabla.index:
            campeon_counts.setdefault(eq, []).append(tabla.loc[eq, 'Campeon'])

    resumen = []
    for eq, vals in campeon_counts.items():
        arr = np.array(vals)
        resumen.append({
            'Equipo': eq,
            'P_Campeon_Media': round(arr.mean(), 1),
            'P_Campeon_Min': round(arr.min(), 1),
            'P_Campeon_Max': round(arr.max(), 1),
            'IC_95_Ancho': round(np.percentile(arr, 97.5) - np.percentile(arr, 2.5), 1),
        })
    return pd.DataFrame(resumen).sort_values('P_Campeon_Media', ascending=False)


# ====================================================================
# MAIN
# ====================================================================

if __name__ == '__main__':

    # ── 0. Modelos ──
    pkls         = ['modelo_goles_L.pkl', 'modelo_1X2_calibrado.pkl', 'columnas_entrenamiento.pkl']
    pkls_existen = all(os.path.exists(p) for p in pkls)

    if FORZAR_REENTRENAMIENTO and pkls_existen:
        print("FORZAR_REENTRENAMIENTO=True — reentrenando...")
        entrenar_modelos()
    elif not pkls_existen:
        print("No se encontraron .pkl — entrenando desde cero...")
        entrenar_modelos()
    else:
        print("Modelos .pkl encontrados — reutilizando.")

    m        = _modelos()
    T_HIST   = m.get('t_optima', 1.0)

    # ── 1. Evaluación en fase de grupos → obtener T óptima del Mundial ──
    T_MUNDIAL = evaluar_fase_grupos(T=T_HIST)

    # ── 2. Cargar datos y construir matriz de cruces con T del Mundial ──
    df_vars, puntos = cargar_datos_mundial()

    print(f"\nMatriz de cruces {len(EQUIPOS_CLASIFICADOS)}×{len(EQUIPOS_CLASIFICADOS)} "
          f"(T_hist={T_HIST:.2f} → T_mundial={T_MUNDIAL:.2f})...")
    mc = matriz_cruces(EQUIPOS_CLASIFICADOS, df_vars, T_MUNDIAL)

    # ── 3. Predicción puntual de eliminatorias ──
    print("\nPredicción puntual de eliminatorias (R32 → Final)...")
    df_elim, campeon = prediccion_eliminatorias(mc, puntos)

    print("\n" + "="*90)
    print("  PREDICCIONES DE ELIMINATORIAS")
    print("="*90)
    cols_mostrar = ['Fase', 'Local', 'Visitante', 'Marcador', 'Avanza',
                    'P_Avanza_L', 'P_Avanza_V', 'Upset', 'Detalle']
    for fase in ['Dieciseisavos', 'Octavos', 'Cuartos', 'Semifinales', '3er Puesto', 'Final']:
        sub = df_elim[df_elim['Fase'] == fase]
        if not sub.empty:
            print(f"\n  ── {fase.upper()} ({FASES_FECHAS.get(fase, '')}) ──")
            print(sub[cols_mostrar].to_string(index=False))

    print(f"\n  ★ CAMPEÓN PREDICHO: {campeon} ★")

    # ── 4. Monte Carlo ──
    print(f"\nMonte Carlo ({N_SIMULACIONES:,} simulaciones desde R32)...")
    tabla_mc = monte_carlo_fase_final(mc, puntos, N_SIMULACIONES)

    print("\n" + "="*90)
    print("  PROBABILIDADES POR FASE (Monte Carlo)")
    print("="*90)
    print(tabla_mc.to_string())

    # ── 5. Intervalos de confianza (bootstrap) ──
    print(f"\nBootstrap de confianza (10 × {N_SIMULACIONES//5:,} sims)...")
    df_ci = bootstrap_confianza(mc, puntos, n_bootstrap=10, n_sims_por_boot=N_SIMULACIONES // 5)
    print("\n  Top 10 — Intervalo de confianza en P(Campeón):")
    print(df_ci.head(10).to_string(index=False))

    # ── 6. Exportar ──
    df_elim.to_csv('Predicciones/predicciones_eliminatorias.csv', index=False, encoding='utf-8-sig')
    tabla_mc.to_csv('Predicciones/probabilidades_montecarlo_fase_final.csv', encoding='utf-8-sig')
    df_ci.to_csv('Predicciones/intervalos_confianza_campeon.csv', index=False, encoding='utf-8-sig')

    # Partidos con mayor potencial de sorpresa
    upsets = df_elim[df_elim['Upset'] >= 50].sort_values('Upset', ascending=False)
    if not upsets.empty:
        print("\n PARTIDOS CON MAYOR POTENCIAL DE SORPRESA (Upset ≥ 50):")
        print(upsets[['Fase', 'Local', 'Visitante', 'P_Avanza_L', 'P_Avanza_V', 'Upset']].to_string(index=False))

    print("\n✓ Archivos exportados en Predicciones/.")