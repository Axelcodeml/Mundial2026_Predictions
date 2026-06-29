import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_predict, KFold
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV

RUTA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RUTA)
os.makedirs('Predicciones', exist_ok=True)

np.random.seed(42)

# ====================================================================
# PARÁMETROS GLOBALES
# ====================================================================

N_SIMULACIONES = 10_000
T_GRUPOS       = 1.0
T_CRUCES       = 1.0
T_FORZADA      = 1
DECAY_RECENCIA = 0.0025
FORZAR_REENTRENAMIENTO = False
N_ITER_REG = 300
N_ITER_1X2 = 800
tweedie_parameter = 1.2
AGRESIVIDAD_GOLES = .6

eq1 = 'Uruguay'
eq2 = 'España'

# ====================================================================
# GRIDS DE HIPERPARÁMETROS
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


def obtener_datos_partido(eq_l, eq_v, df_vars):
    # Extraemos las filas y forzamos a que sean series planas
    stats_l = df_vars[df_vars['Equipo'] == eq_l].iloc[0].add_suffix('_Local')
    stats_v = df_vars[df_vars['Equipo'] == eq_v].iloc[0].add_suffix('_Visitante')
    
    # Creamos el registro base
    data = {'Fecha': '2026-07-01', 'Equipo_Local': eq_l, 'Equipo_Visitante': eq_v, 'Grupo': 'A'}
    data.update(stats_l.to_dict())
    data.update(stats_v.to_dict())
    
    # Retornamos el DataFrame con una sola fila
    return pd.DataFrame([data])


def analizar_partido(eq_l, eq_v, df_vars):
    # 1. Obtener datos base
    df_partido = obtener_datos_partido(eq_l, eq_v, df_vars)
    m = _modelos()
    
    # 2. PROCESAMIENTO ÚNICO (La "Fuente de la Verdad")
    # Copiamos exactamente la lógica de tu pipeline para evitar discrepancias
    df_calc = df_partido.copy()
    
    # Replicamos el cálculo de diferencias del pipeline
    cols_avg = [c for c in df_calc.columns
                if c.endswith(('_2_Local', '_3_Local', '_5_Local', '_total_Local'))
                and c.startswith('avg_')]
    for col_local in cols_avg:
        col_v = col_local.replace('_Local', '_Visitante')
        if col_v in df_calc.columns:
            nombre_base = col_local.replace('_Local', '')
            df_calc[f'diff_{nombre_base}'] = df_calc[col_local] - df_calc[col_v]

    # Variables especiales (igual que pipeline_prediccion)
    for sufijo in ['trend_xG_5', 'avg_xGA_2', 'avg_xGA_5', 'avg_xGA_total',
                    'clean_sheet_rate_5', 'forma_vs_historia']:
        col_l = f'{sufijo}_Local'
        col_v_s = f'{sufijo}_Visitante'
        if col_l in df_calc.columns and col_v_s in df_calc.columns:
            df_calc[f'diff_{sufijo}'] = df_calc[col_l] - df_calc[col_v_s]
            
    df_calc['diff_Puntos'] = df_calc['avg_Puntos_total_Local'] - df_calc['avg_Puntos_total_Visitante']
    df_calc['Prob_Implicita_ELO'] = 1 / (1 + 10 ** (-df_calc['diff_Puntos'] / 400))
    df_calc['diff_Valor_Mercado'] = df_calc['Valor_Mercado_Millones_Eur_Local'] - df_calc['Valor_Mercado_Millones_Eur_Visitante']
    
    # Aseguramos formato numérico
    for col in df_calc.select_dtypes(include=[np.number]).columns:
        df_calc[col] = df_calc[col].fillna(0)

    # 3. Preparar entrada para modelos (Alineación perfecta con m['cols'])
    for col in m['cols']:
        if col not in df_calc.columns: df_calc[col] = 0
    X_final = df_calc[m['cols']]

    # 4. Inferencia conjunta
    g_l = m['L'].predict(X_final)[0]
    g_v = m['V'].predict(X_final)[0]
    
    # Probabilidades (usando el pipeline que ya tenemos ajustado)
    res = pipeline_prediccion(df_partido, sede_neutral=True)
    
    # 5. Imprimir informe unificado
    print(f"\n{'='*50}\nPREDICCIONES: {eq_l} vs {eq_v}\n{'='*50}")
    print(f"Goles Predichos: {g_l:.2f} - {g_v:.2f}")
    print(f"Probabilidades (1-X-2): {res['Prob_Local'].iloc[0]:.1%} - {res['Prob_Empate'].iloc[0]:.1%} - {res['Prob_Visitante'].iloc[0]:.1%}")
    
    print(f"\n--- VARIABLES CLAVE (Top 20) ---")
    df_imp = pd.read_csv('Predicciones/importancia_variables_top25.csv')
    for var in df_imp['Variable'].head(20):
        if var in df_calc.columns:
            print(f"{var:<40} : {df_calc[var].iloc[0]:.3f}")

# ====================================================================
# 1. ENTRENAMIENTO
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

    print("\n--- RESULTADOS EN TEST (15% temporal) ---")
    print(classification_report(y_test_1X2, pred_test_clases))
    print(f"Accuracy test: {accuracy_score(y_test_1X2, pred_test_clases):.3f}")

    print("\nOptimizando Temperatura (T)...")
    t_optimo = tunear_temperatura(probs_test, y_test_1X2, name="Histórico Global")

    print("\nFase 3: Reentrenando con el 100% de los datos...")
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

    # --- Importancia de las variables Modelo Final --- #
    print("\nTop 25 variables más importantes (Modelo 1X2 Final):")
    # Las columnas del modelo 1X2 son las base + las dos predicciones de la Fase 1
    columnas_1X2 = list(X.columns) + ['Pred_Goles_L', 'Pred_Goles_V']
    
    # Sacamos las importancias del modelo final (no del modelo_L)
    importancias_1X2 = modelo_1X2_final.feature_importances_
    
    df_imp = pd.DataFrame({'Variable': columnas_1X2, 'Importancia': importancias_1X2})
    df_imp = df_imp.sort_values('Importancia', ascending=False).head(25)
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

    print("\nModelos guardados. ¡Listo para el Mundial!")


# ====================================================================
# 2. PIPELINE DE PREDICCIÓN
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
# 3. CARGA DEL MUNDIAL
# ====================================================================

def cargar_mundial():
    df_mundial_all = pd.read_csv('./Data/partidos_mundial.csv')
    df_mundial_all['Jugado'] = df_mundial_all['Jugado'].astype(str).str.lower() == 'true'

    # Dedup de seguridad: priorizar la versión jugada si hay duplicados
    df_mundial_all = (
        df_mundial_all
        .sort_values('Jugado', ascending=True)
        .drop_duplicates(subset=['Equipo_Local', 'Equipo_Visitante', 'Fecha'], keep='last')
        .reset_index(drop=True)
    )

    df_mundial = df_mundial_all[['Fecha', 'Equipo_Local', 'Equipo_Visitante']].copy()
    df_vars = pd.read_csv('./Data/datos_mundial.csv').sort_values('Fecha')
    # Un solo snapshot por equipo
    df_vars = df_vars.drop_duplicates('Equipo', keep='last').reset_index(drop=True)
    grupos  = pd.read_csv('./Data/Grupos_Mundial.csv', sep=';')

    # --- LIMPIEZA DE SEGURIDAD ---
    # Identificamos todas las columnas que deberían ser numéricas
    cols_numericas = [c for c in df_vars.columns if c not in ['Equipo', 'Fecha', 'Grupo', 'Continente']]
    for col in cols_numericas:
        # Convertimos a numérico, forzando error a NaN, y rellenamos NaN con 0
        df_vars[col] = pd.to_numeric(df_vars[col], errors='coerce').fillna(0)

    aux = df_mundial.merge(df_vars, left_on='Equipo_Local',    right_on='Equipo', how='left')
    df_mundial_vars = aux.merge(
        df_vars, left_on='Equipo_Visitante', right_on='Equipo', how='left',
    )
    df_mundial_vars.drop(
        columns=[c for c in df_mundial_vars.columns
                 if c in ('Equipo_x', 'Equipo_y', 'Fecha_x', 'Fecha_y',
                           'Resultado_1X2_x', 'Resultado_1X2_y',
                           'Tipo_Equipo_x', 'Tipo_Equipo_y')],
        inplace=True, errors='ignore',
    )
    df_mundial_vars.columns = df_mundial_vars.columns.str.replace(r'_x$', '_Local',    regex=True)
    df_mundial_vars.columns = df_mundial_vars.columns.str.replace(r'_y$', '_Visitante', regex=True)

    df_mundial_grupos = pd.merge(
        df_mundial_vars, grupos, left_on='Equipo_Local', right_on='Equipo',
    ).drop('Equipo', axis=1)

    # ── Añadir info de partidos jugados ──
    jugado_info = df_mundial_all[df_mundial_all['Jugado']][
        ['Equipo_Local', 'Equipo_Visitante', 'Goles_Local', 'Goles_Visitante', 'Resultado_1X2']
    ].rename(columns={
        'Goles_Local': 'Goles_Local_Real',
        'Goles_Visitante': 'Goles_Visitante_Real',
        'Resultado_1X2': 'Resultado_1X2_Real',
    })

    if not jugado_info.empty:
        df_mundial_grupos = df_mundial_grupos.merge(
            jugado_info, on=['Equipo_Local', 'Equipo_Visitante'], how='left'
        )
        df_mundial_grupos['Jugado'] = df_mundial_grupos['Goles_Local_Real'].notna()
    else:
        df_mundial_grupos['Jugado'] = False
        df_mundial_grupos['Goles_Local_Real'] = np.nan
        df_mundial_grupos['Goles_Visitante_Real'] = np.nan
        df_mundial_grupos['Resultado_1X2_Real'] = ''

    fechas_reales = df_mundial_all[['Fecha', 'Equipo_Local', 'Equipo_Visitante']]
    return df_mundial_grupos, df_vars, grupos, fechas_reales


# ====================================================================
# 4. MATRIZ DE CRUCES 48x48
# ====================================================================

def matriz_cruces(equipos, df_vars, T_cruces):
    # CORRECCIÓN VITAL: Quitamos 'Fecha' de df_vars antes del merge para evitar colisiones 
    # de columnas (Fecha_x y Fecha_y) que romperían el pipeline_prediccion.
    df_vars_clean = df_vars.drop(columns=['Fecha'], errors='ignore')

    pares       = [(a, b) for a in equipos for b in equipos if a != b]
    partido     = pd.DataFrame({
        'Fecha':            ['2026-07-01'] * len(pares),
        'Equipo_Local':    [a for a, b in pares],
        'Equipo_Visitante': [b for a, b in pares],
    })
    partido_vars = partido.merge(df_vars_clean, left_on='Equipo_Local',    right_on='Equipo', how='left').drop(columns=['Equipo'])
    partido_vars = partido_vars.merge(df_vars_clean, left_on='Equipo_Visitante', right_on='Equipo', how='left',
                                      suffixes=('_Local', '_Visitante')).drop(columns=['Equipo'])
    
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

    M_adv = P1 + PX * (P1 / np.where(P1 + P2 == 0, 1, P1 + P2))
    np.fill_diagonal(M_adv, 0.5)
    return {'P1': P1, 'PX': PX, 'P2': P2, 'XGL': XGL, 'XGV': XGV, 'M_adv': M_adv, 'idx': idx}


# ====================================================================
# 5. UTILIDADES DE PREDICCIÓN PUNTUAL
# ====================================================================

def _poisson_pmf(k, lam):
    lam = max(lam, 0.05)
    return math.exp(-lam) * lam ** k / math.factorial(k)


def marcador_mas_probable(xg_l, xg_v, prob_1, prob_x, prob_2, resultado, agresividad=AGRESIVIDAD_GOLES):
    """
    Calcula el marcador usando Poisson, inflando los xG dinámicamente
    en base a la probabilidad real (0.0 a 1.0) de victoria de cada equipo.
    """
    # ESCALADO DINÁMICO: Multiplicamos el xG base por (1 + probabilidad * agresividad)
    # Ejemplo: Si prob_1 es 0.90 (90%), su xG se multiplica por (1 + 0.90*1.2) = 2.08 (¡Sube más del doble!)
    # Si prob_2 es 0.20 (20%), su xG se multiplica por (1 + 0.20*1.2) = 1.24 (Sube un poco por agresividad)
    lam_l = xg_l
    lam_v = xg_v
    if prob_1 > prob_2:
        lam_l = max(xg_l * (1 + (prob_1 * agresividad)), 0.1)
        lam_v = max(xg_v * (1 + (prob_2 * agresividad)/4), 0.1)
    if prob_2 > prob_1:
        lam_l = max(xg_l * (1 + (prob_1 * agresividad/4)), 0.1)
        lam_v = max(xg_v * (1 + (prob_2 * agresividad)), 0.1)
    
    # TRATAMIENTO DEL EMPATE: Si el resultado predicho es 'X', empujamos ambos xG 
    # en función de lo fuerte que sea esa probabilidad de empate (para evitar el 0-0 eterno)
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
# 6. CLASIFICACIÓN DE GRUPOS
# ====================================================================

CRUCES_R32 = [
    ('1E', '3_1'), ('1I', '3_2'), ('2A', '2B'), ('1F', '2C'),
    ('2K', '2L'), ('1H', '2J'), ('1D', '3_3'), ('1G', '3_4'),
    ('1C', '2F'), ('2E', '2I'), ('1A', '3_5'), ('1L', '3_6'),
    ('1J', '2H'), ('2D', '2G'), ('1B', '3_7'), ('1K', '3_8'),
]


def clasificar_grupos(df, tiradas):
    df  = df.copy()
    tir = np.asarray(tiradas)
    df['Pts_L'] = np.where(tir == 0, 3, np.where(tir == 1, 1, 0))
    df['Pts_V'] = np.where(tir == 2, 3, np.where(tir == 1, 1, 0))

    locales    = df[['Grupo', 'Equipo_Local',    'Pts_L', 'xG_Modelo_Local',    'xG_Modelo_Visitante']].rename(
        columns={'Equipo_Local':    'Equipo', 'Pts_L': 'Pts', 'xG_Modelo_Local':    'GF', 'xG_Modelo_Visitante': 'GC'})
    visitantes = df[['Grupo', 'Equipo_Visitante', 'Pts_V', 'xG_Modelo_Visitante', 'xG_Modelo_Local']].rename(
        columns={'Equipo_Visitante': 'Equipo', 'Pts_V': 'Pts', 'xG_Modelo_Visitante': 'GF', 'xG_Modelo_Local': 'GC'})

    clasif = pd.concat([locales, visitantes])
    clasif['DG'] = clasif['GF'] - clasif['GC']
    clasif = clasif.groupby(['Grupo', 'Equipo']).sum().reset_index()
    clasif = clasif.sort_values(['Grupo', 'Pts', 'DG', 'GF'], ascending=[True, False, False, False])
    clasif['Posicion'] = clasif.groupby('Grupo').cumcount() + 1

    terceros = clasif[clasif['Posicion'] == 3].sort_values(['Pts', 'DG', 'GF'], ascending=False).head(8)
    pos = {f"{r['Posicion']}{r['Grupo']}": r['Equipo'] for _, r in clasif[clasif['Posicion'] <= 2].iterrows()}
    for i, eq in enumerate(terceros['Equipo']): pos[f'3_{i + 1}'] = eq
    return clasif, terceros, pos


# ====================================================================
# 7. MONTE CARLO VECTORIZADO
# ====================================================================

def monte_carlo(df_pred_grupos, mc, equipos, n_sims=N_SIMULACIONES):
    idx  = mc['idx']
    n_eq = len(equipos)
    df   = df_pred_grupos

    eq_grupo = pd.concat([
        df[['Grupo', 'Equipo_Local']].rename(columns={'Equipo_Local': 'Equipo'}),
        df[['Grupo', 'Equipo_Visitante']].rename(columns={'Equipo_Visitante': 'Equipo'}),
    ]).drop_duplicates()
    grupos_de = {r['Equipo']: r['Grupo'] for _, r in eq_grupo.iterrows()}
    letras    = sorted(eq_grupo['Grupo'].unique())
    miembros  = {g: sorted([e for e, gg in grupos_de.items() if gg == g]) for g in letras}

    loc_series = df['Equipo_Local'].map(idx)
    vis_series = df['Equipo_Visitante'].map(idx)

    if loc_series.isna().any() or vis_series.isna().any():
        raise ValueError(
            f"Equipos sin índice — revisa Grupos_Mundial.csv.\n"
            f"  Locales: {df[loc_series.isna()]['Equipo_Local'].unique()}\n"
            f"  Visitantes: {df[vis_series.isna()]['Equipo_Visitante'].unique()}"
        )

    loc = loc_series.astype(int).to_numpy()
    vis = vis_series.astype(int).to_numpy()

    P    = df[['Prob_Local', 'Prob_Empate', 'Prob_Visitante']].to_numpy()
    P    = P / P.sum(axis=1, keepdims=True)
    xg_l = df['xG_Modelo_Local'].to_numpy()
    xg_v = df['xG_Modelo_Visitante'].to_numpy()

    gf = np.zeros(n_eq); gc = np.zeros(n_eq)
    np.add.at(gf, loc, xg_l); np.add.at(gf, vis, xg_v)
    np.add.at(gc, loc, xg_v); np.add.at(gc, vis, xg_l)
    dg   = gf - gc
    dg_c = np.round(dg * 100).astype(np.int64)
    gf_c = np.round(gf * 100).astype(np.int64)

    u    = np.random.random((n_sims, len(df)))
    cum1 = P[:, 0]; cum2 = P[:, 0] + P[:, 1]
    out  = np.where(u < cum1, 0, np.where(u < cum2, 1, 2)).astype(np.int8)

    pts = np.zeros((n_sims, n_eq), dtype=np.int64)
    for m_idx in range(len(df)):
        o = out[:, m_idx]
        pts[:, loc[m_idx]] += np.where(o == 0, 3, np.where(o == 1, 1, 0))
        pts[:, vis[m_idx]] += np.where(o == 2, 3, np.where(o == 1, 1, 0))

    clave = pts * 10_000_000_000 + (dg_c[None, :] + 100_000) * 10_000 + gf_c[None, :]

    pos_eq         = {}
    terceros_ids   = np.zeros((n_sims, 12), dtype=np.int64)
    terceros_clave = np.zeros((n_sims, 12), dtype=np.int64)
    for gi, g in enumerate(letras):
        ids      = np.array([idx[e] for e in miembros[g]])
        k        = clave[:, ids]
        orden    = np.argsort(-k, axis=1, kind='stable')
        ordenados = ids[orden]
        pos_eq[f'1{g}']       = ordenados[:, 0]
        pos_eq[f'2{g}']       = ordenados[:, 1]
        terceros_ids[:, gi]   = ordenados[:, 2]
        terceros_clave[:, gi] = np.take_along_axis(k, orden[:, 2:3], axis=1)[:, 0]

    orden_terceros = np.argsort(-terceros_clave, axis=1, kind='stable')
    mejores8       = np.take_along_axis(terceros_ids, orden_terceros[:, :8], axis=1)
    for r in range(8): pos_eq[f'3_{r + 1}'] = mejores8[:, r]

    M          = mc['M_adv']
    contadores = {f: np.zeros(n_eq, dtype=np.int64)
                  for f in ['R32', 'Octavos', 'Cuartos', 'Semis', 'Final', 'Campeon']}

    clasificados = np.stack(
        [pos_eq[f'1{g}'] for g in letras] +
        [pos_eq[f'2{g}'] for g in letras] +
        [mejores8[:, r]  for r in range(8)],
        axis=1,
    )
    for c in range(32): np.add.at(contadores['R32'], clasificados[:, c], 1)

    def jugar(a_ids, b_ids):
        p      = M[a_ids, b_ids]
        gana_a = np.random.random(a_ids.shape) < p
        return np.where(gana_a, a_ids, b_ids)

    r32 = [(pos_eq[a], pos_eq[b]) for a, b in CRUCES_R32]
    g32 = [jugar(a, b) for a, b in r32]
    for w in g32: np.add.at(contadores['Octavos'], w, 1)
    g16 = [jugar(g32[i], g32[i + 1]) for i in range(0, 16, 2)]
    for w in g16: np.add.at(contadores['Cuartos'], w, 1)
    gqf = [jugar(g16[i], g16[i + 1]) for i in range(0, 8, 2)]
    for w in gqf: np.add.at(contadores['Semis'], w, 1)
    sf1 = jugar(gqf[0], gqf[1]); sf2 = jugar(gqf[2], gqf[3])
    for w in (sf1, sf2): np.add.at(contadores['Final'], w, 1)
    campeon = jugar(sf1, sf2)
    np.add.at(contadores['Campeon'], campeon, 1)

    tabla = pd.DataFrame(
        {f: contadores[f] / n_sims * 100 for f in contadores}, index=equipos,
    )
    return tabla.sort_values(['Campeon', 'Final', 'Semis'], ascending=False).round(1)


# ====================================================================
# 8. PREDICCIÓN PUNTUAL DE TODO EL MUNDIAL
# ====================================================================

FASES_FECHAS = {
    'Dieciseisavos': '28 jun - 3 jul', 'Octavos': '4 - 7 jul', 'Cuartos': '9 - 11 jul',
    'Semifinales':   '14 - 15 jul',    '3er Puesto': '18 jul', 'Final': '19 jul',
}


def prediccion_puntual(df_pred_grupos, mc, equipos, fechas_reales):
    idx     = mc['idx']
    probs   = df_pred_grupos[['Prob_Local', 'Prob_Empate', 'Prob_Visitante']].to_numpy()
    tiradas = probs.argmax(axis=1)

    filas       = []
    mapa_fechas = {(r['Equipo_Local'], r['Equipo_Visitante']): r['Fecha']
                   for _, r in fechas_reales.iterrows()}

    # BUCLE 1: FASE DE GRUPOS
    for k, (_, r) in enumerate(df_pred_grupos.iterrows()):
        jugado = r.get('Jugado', False)

        if jugado:
            # Resultado real — usar marcador real directamente
            gl = int(r['Goles_Local_Real'])
            gv = int(r['Goles_Visitante_Real'])
            res = '1' if gl > gv else ('X' if gl == gv else '2')
        else:
            # Predicción
            res    = ['1', 'X', '2'][tiradas[k]]
            gl, gv = marcador_mas_probable(
                r['xG_Modelo_Local'], r['xG_Modelo_Visitante'], 
                r['Prob_Local'], r['Prob_Empate'], r['Prob_Visitante'], 
                res
            )
        
        filas.append({
            'Fecha':             mapa_fechas.get((r['Equipo_Local'], r['Equipo_Visitante']), ''),
            'Grupo':             r['Grupo'],
            'Local':             r['Equipo_Local'],
            'Visitante':         r['Equipo_Visitante'],
            'Marcador': f'{gl}-{gv}',
            'Resultado':     res,
            'xG_L':              round(r['xG_Modelo_Local'],    2),
            'xG_V':              round(r['xG_Modelo_Visitante'], 2),
            'Prob_1':            round(r['Prob_Local']    * 100, 1),
            'Prob_X':            round(r['Prob_Empate']   * 100, 1),
            'Prob_2':            round(r['Prob_Visitante'] * 100, 1),
            'Tipo':              'Real' if jugado else 'Predicción',
        })

    df_grupos_pred = pd.DataFrame(filas).sort_values(['Grupo', 'Fecha']).reset_index(drop=True)
    clasif, terceros, pos = clasificar_grupos(df_pred_grupos, tiradas)

    P1, PX, P2, XGL, XGV = mc['P1'], mc['PX'], mc['P2'], mc['XGL'], mc['XGV']
    registro = []

    # BUCLE 2: ELIMINATORIAS (CRUCES)
    def jugar_cruce(fase, eq_a, eq_b):
        i, j       = idx[eq_a], idx[eq_b]
        p1, px, p2 = P1[i, j], PX[i, j], P2[i, j]
        res        = ['1', 'X', '2'][int(np.argmax([p1, px, p2]))]
        
        if res == 'X':
            ganador  = eq_a if p1 >= p2 else eq_b
            # Pasamos p1, px, p2 a la función en caso de empate
            gl, gv   = marcador_mas_probable(XGL[i, j], XGV[i, j], p1, px, p2, 'X')
            marcador = f'{gl}-{gv} (pen)'
            detalle  = 'Empate; gana en penaltis'
        else:
            ganador  = eq_a if res == '1' else eq_b
            # Pasamos p1, px, p2 a la función en caso de victoria clara
            gl, gv   = marcador_mas_probable(XGL[i, j], XGV[i, j], p1, px, p2, res)
            marcador = f'{gl}-{gv}'
            detalle  = 'Tiempo regular'
            
        registro.append({
            'Fase': fase, 'Fechas': FASES_FECHAS[fase],
            'Local': eq_a, 'Visitante': eq_b,
            'Marcador': marcador, 'Avanza': ganador,
            'xG_L': round(XGL[i, j], 2), 'xG_V': round(XGV[i, j], 2),
            'Prob_1': round(p1 * 100, 1), 'Prob_X': round(px * 100, 1),
            'Prob_2': round(p2 * 100, 1), 'Detalle': detalle,
        })
        return ganador

    g32 = [jugar_cruce('Dieciseisavos', pos[a], pos[b]) for a, b in CRUCES_R32]
    g16 = [jugar_cruce('Octavos',       g32[i], g32[i + 1]) for i in range(0, 16, 2)]
    gqf = [jugar_cruce('Cuartos',       g16[i], g16[i + 1]) for i in range(0, 8,  2)]
    sf  = [jugar_cruce('Semifinales', gqf[0], gqf[1]),
           jugar_cruce('Semifinales', gqf[2], gqf[3])]
    perdedores_sf = [e for par in [(gqf[0], gqf[1]), (gqf[2], gqf[3])] for e in par if e not in sf]
    jugar_cruce('3er Puesto', perdedores_sf[0], perdedores_sf[1])
    campeon = jugar_cruce('Final', sf[0], sf[1])

    return df_grupos_pred, clasif, terceros, pos, pd.DataFrame(registro), campeon


# ====================================================================
# MAIN
# ====================================================================

if __name__ == '__main__':
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
    T_OPTIMA = m.get('t_optima', 1.0)
    T_GRUPOS = T_OPTIMA * T_FORZADA
    T_CRUCES = T_OPTIMA * T_FORZADA

    print(f"\nFase de grupos (T={T_GRUPOS})...")
    df_mundial_grupos, df_vars, grupos, fechas_reales = cargar_mundial()

    # Columnas extra que NO van al pipeline de predicción
    _cols_jugado = ['Jugado', 'Goles_Local_Real', 'Goles_Visitante_Real', 'Resultado_1X2_Real']
    df_predicciones_mundial = pipeline_prediccion(
        df_mundial_grupos.drop(columns=_cols_jugado, errors='ignore'),
        sede_neutral=True, T=T_GRUPOS,
    )
    df_predicciones_mundial['Grupo'] = df_mundial_grupos['Grupo'].values

    # ── Sobreescribir partidos ya jugados con resultados reales ──
    mask_jugado = df_mundial_grupos['Jugado'].values
    if mask_jugado.any():
        df_predicciones_mundial.loc[mask_jugado, 'xG_Modelo_Local'] = (
            df_mundial_grupos.loc[mask_jugado, 'Goles_Local_Real'].values.astype(float)
        )
        df_predicciones_mundial.loc[mask_jugado, 'xG_Modelo_Visitante'] = (
            df_mundial_grupos.loc[mask_jugado, 'Goles_Visitante_Real'].values.astype(float)
        )
        res_real = df_mundial_grupos.loc[mask_jugado, 'Resultado_1X2_Real'].values.astype(str)
        df_predicciones_mundial.loc[mask_jugado, 'Prob_Local']     = (res_real == '1').astype(float)
        df_predicciones_mundial.loc[mask_jugado, 'Prob_Empate']    = (res_real == 'X').astype(float)
        df_predicciones_mundial.loc[mask_jugado, 'Prob_Visitante'] = (res_real == '2').astype(float)
        print(f"  → {mask_jugado.sum()} partidos ya jugados (resultados reales inyectados)")

    # Flags para uso posterior
    df_predicciones_mundial['Jugado'] = mask_jugado
    df_predicciones_mundial['Goles_Local_Real'] = df_mundial_grupos['Goles_Local_Real'].values
    df_predicciones_mundial['Goles_Visitante_Real'] = df_mundial_grupos['Goles_Visitante_Real'].values

    equipos = sorted(grupos['Equipo'].unique())
    print(f"Matriz de cruces 48×48 (T={T_CRUCES})...")
    mc = matriz_cruces(equipos, df_vars, T_CRUCES)

    print("Predicción puntual de los 104 partidos...")
    df_grupos_pred, clasif, terceros, pos, df_elim, campeon = prediccion_puntual(
        df_predicciones_mundial, mc, equipos, fechas_reales,
    )

    print(f"Monte Carlo ({N_SIMULACIONES} simulaciones)...")
    tabla_mc = monte_carlo(df_predicciones_mundial, mc, equipos, N_SIMULACIONES)

    df_grupos_pred.sort_values('Fecha', inplace=True)
    df_grupos_pred.to_csv('Predicciones/predicciones_fase_grupos.csv',    index=False, encoding='utf-8-sig')
    clasif[['Grupo', 'Equipo', 'Pts', 'GF', 'GC', 'DG', 'Posicion']].to_csv(
        'Predicciones/clasificacion_grupos.csv', index=False, encoding='utf-8-sig')
    df_elim.to_csv('Predicciones/predicciones_eliminatorias.csv',         index=False, encoding='utf-8-sig')
    tabla_mc.to_csv('Predicciones/probabilidades_montecarlo.csv',          encoding='utf-8-sig')

    print(f"\n=== CAMPEÓN PREDICHO: {campeon} ===")
    print(f"\nTop 20 por probabilidad de campeón (Monte Carlo, {N_SIMULACIONES} sims):")
    print(tabla_mc.head(20).to_string())
    print("\nArchivos escritos en Predicciones/.")

    # ─────────────────────────────────────────────────────────────
    # IMPRESIÓN POR CONSOLA DE LA FASE DE GRUPOS
    # ─────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("=== FASE DE GRUPOS (Reales + Predicciones) ===")
    print("="*80)
    columnas_a_mostrar = ['Tipo', 'Grupo', 'Fecha', 'Local', 'Visitante', 'Marcador', 'xG_L', 'xG_V', 'Prob_1', 'Prob_X', 'Prob_2']
    columnas_a_mostrar = [c for c in columnas_a_mostrar if c in df_grupos_pred.columns]
    df_grupos_pred['Fecha'] = pd.to_datetime(df_grupos_pred['Fecha'])

    # Partidos ya jugados
    if 'Tipo' in df_grupos_pred.columns:
        df_jugados_print = df_grupos_pred[df_grupos_pred['Tipo'] == 'Real']
        if not df_jugados_print.empty:
            print("\n── RESULTADOS REALES ──")
            print(df_jugados_print[columnas_a_mostrar].to_string(index=False))

        # Partidos pendientes (predicciones)
        df_pendientes_print = df_grupos_pred[df_grupos_pred['Tipo'] == 'Predicción']
        if not df_pendientes_print.empty:
            print("\n── PREDICCIONES PENDIENTES ──")
            print(df_pendientes_print[columnas_a_mostrar].to_string(index=False))
    else:
        print(df_grupos_pred[columnas_a_mostrar].to_string(index=False))

    # Calcula la media de goles por partido a partir de la columna 'Marcador_Predicho' con formato 'G1-G2'
    try:
        goles_split = df_grupos_pred['Marcador'].str.split('-', expand=True)
        goles_split = goles_split.astype(float)
        goles_totales = goles_split.iloc[:, 0].fillna(0) + goles_split.iloc[:, 1].fillna(0)
        media_goles = goles_totales.mean().round(2)
    except Exception:
        # En caso de datos mal formados, fallback a NaN
        media_goles = float('nan')

    print("=== PREDICCIONES GOLES ===")
    df_historico_goles = pd.read_csv('./Data/datos_historicos.csv')
    print(f"Media de goles por partido historica: {(df_historico_goles['Goles_Local'] + df_historico_goles['Goles_Visitante']).mean().round(2)}")
    print(f"Media de goles por partido predicciones fase de grupos: {media_goles}")
    print(f"Media de xG por partido predicciones fase de grupos: {(df_grupos_pred['xG_L'] + df_grupos_pred['xG_V']).mean().round(2)}")


    # 2. PREDICCIÓN PUNTUAL (Si quieres probar un partido, descomenta esto)
    analizar_partido(eq1, eq2, df_vars)