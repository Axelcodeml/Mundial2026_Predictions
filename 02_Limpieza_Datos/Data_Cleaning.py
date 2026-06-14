import re
import os
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# CONFIGURACIÓN
# ============================================================
RUTA_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RUTA_RAIZ)

RUTA_PARTIDOS      = './Data/partidos.csv'
RUTA_RANKING       = './Data/ranking_fifa.csv'
RUTA_TRANSFERMARKT = './Data/transfermarkt.csv'
RUTA_OUT_HISTORICO = './Data/datos_historicos.csv'
RUTA_OUT_MUNDIAL   = './Data/datos_mundial.csv'
RUTA_OUT_PARTIDOS  = './Data/partidos_mundial.csv'

FECHA_CORTE   = pd.to_datetime('2026-06-11')
FECHA_MINIMA  = pd.to_datetime('2021-01-01')   

VENTANAS = [3, 5]   

equipos_mundial = [
    'Alemania', 'Austria', 'Bélgica', 'Bosnia-Herzegovina', 'Croacia',
    'Escocia', 'España', 'Francia', 'Inglaterra', 'Noruega',
    'Países Bajos', 'Portugal', 'República Checa', 'Suecia', 'Suiza', 'Turquía',
    'Argentina', 'Brasil', 'Colombia', 'Ecuador', 'Paraguay', 'Uruguay',
    'Canadá', 'Curazao', 'EE. UU.', 'Haití', 'México', 'Panamá',
    'Argelia', 'Cabo Verde', 'Costa de Marfil', 'Egipto', 'Ghana',
    'Marruecos', 'RD Congo', 'Senegal', 'Sudáfrica', 'Túnez',
    'Arabia Saudí', 'Australia', 'Catar', 'Corea del Sur', 'Irak',
    'Irán', 'Japón', 'Jordania', 'Uzbekistán', 'Nueva Zelanda',
]

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

# ============================================================
# 1. CARGA Y JOINS
# ============================================================

print("Cargando datos...")
df               = pd.read_csv(RUTA_PARTIDOS)
df_ranking       = pd.read_csv(RUTA_RANKING)
df_transfermarkt = pd.read_csv(RUTA_TRANSFERMARKT)

df.drop_duplicates(subset=['URL', 'Fecha', 'Equipo_Local', 'Equipo_Visitante'], keep='last', inplace=True)
df = df[df['Equipo_Local'].isin(equipos_mundial) | df['Equipo_Visitante'].isin(equipos_mundial)]

df.loc[
    (df['Equipo_Local'] == 'Bolivia') & (df['Fecha'] == '11.06.2026 02:00'), 'Fecha'
] = '10.06.2026 02:00'

mapeo_paises = {
    'EEUU': 'EE. UU.', 'RI de Irán': 'Irán', 'República de Corea': 'Corea del Sur',
    'RD del Congo': 'RD Congo', 'Bosnia y Herzegovina': 'Bosnia-Herzegovina',
    'Baréin': 'Bahréin', 'RP China': 'China', 'República Kirguisa': 'Kirguistán',
    'Kazajstán': 'Kazajistán', 'RDP de Corea': 'Corea del Norte',
    'Guinea-Bissáu': 'Guinea-Bisáu', 'San Cristóbal y Nieves': 'Saint Kitts y Nevis',
    'Hong Kong, China': 'Hong Kong', 'Myanmar': 'Birmania', 'Esuatini': 'Eswatini',
    'Chinese Taipei': 'China Taipei', 'Bangladesh': 'Bangladés',
    'Samoa Estadounidense': 'Samoa Americana', 'Brunéi Darussalam': 'Brunéi',
    'República de Irlanda': 'Irlanda', 'Qatar': 'Catar',
    'Turcas y Caicos': 'Islas Turcas y Caicos',
    'Antigua y Barbuda': 'Antigua & Barbuda', 'Comoras': 'Comores',
}
df_ranking['País'] = df_ranking['País'].replace(mapeo_paises)

df['Fecha']         = pd.to_datetime(df['Fecha'], format='%d.%m.%Y %H:%M').dt.normalize()
df['Fecha']         = pd.to_datetime(df['Fecha']).dt.tz_localize(None)
df_ranking['Fecha'] = pd.to_datetime(df_ranking['Fecha'], format='%d-%m-%Y').dt.tz_localize(None)

df         = df.sort_values('Fecha').reset_index(drop=True)
df_ranking = df_ranking.sort_values('Fecha').reset_index(drop=True)

df = pd.merge_asof(
    df, df_ranking, on='Fecha', left_by='Equipo_Local', right_by='País',
    direction='backward', allow_exact_matches=False
).rename(columns={'Puntuación': 'Puntos_Local'})

df = pd.merge_asof(
    df, df_ranking, on='Fecha', left_by='Equipo_Visitante', right_by='País',
    direction='backward', allow_exact_matches=False
).rename(columns={'Puntuación': 'Puntos_Visitante'})

df.drop(columns=[c for c in df.columns if 'País' in c], inplace=True)

# --- Valor de Mercado ---
diccionario_Transfermarkt = {
    'Estados Unidos': 'EE. UU.', 'Chequia': 'República Checa',
    'República Democrática del Congo': 'RD Congo', 'Arabia Saudita': 'Arabia Saudí',
    'Malí': 'Mali', 'Comoras': 'Comores', 'Baréin': 'Bahréin', 'Myanmar': 'Birmania',
    'República del Congo': 'Congo', 'China Taipéi': 'China Taipei',
    'Esuatini': 'Eswatini', 'Antigua y Barbuda': 'Antigua & Barbuda',
    'San Cristóbal y Nieves': 'Saint Kitts y Nevis', 'Brunéi Darussalam': 'Brunéi',
    'Islas Vírgenes de los Estados Unidos': 'Islas Vírgenes Estadounidenses',
}
df_transfermarkt['Equipo'] = df_transfermarkt['Equipo'].replace(diccionario_Transfermarkt)
df_transfermarkt.drop('Valor_Mercado_Raw', axis=1, inplace=True, errors='ignore')

df = df.merge(df_transfermarkt, left_on='Equipo_Local',    right_on='Equipo', how='left') \
       .rename(columns={'Valor_Mercado_Millones_Eur': 'Valor_Mercado_Millones_Eur_Local'}) \
       .drop(columns=['Equipo'])
df = df.merge(df_transfermarkt, left_on='Equipo_Visitante', right_on='Equipo', how='left') \
       .rename(columns={'Valor_Mercado_Millones_Eur': 'Valor_Mercado_Millones_Eur_Visitante'}) \
       .drop(columns=['Equipo'])

df['Valor_Mercado_Millones_Eur_Local'] = df['Valor_Mercado_Millones_Eur_Local'].fillna(0.0)
df['Valor_Mercado_Millones_Eur_Visitante'] = df['Valor_Mercado_Millones_Eur_Visitante'].fillna(0.0)

# --- Continente y Peso ---
df['Continente_Local']     = df['Equipo_Local'].map(mapa_continentes)
df['Continente_Visitante'] = df['Equipo_Visitante'].map(mapa_continentes)
df['Peso_Local']           = df['Continente_Local'].map(pesos_continente)
df['Peso_Visitante']       = df['Continente_Visitante'].map(pesos_continente)
df.drop(['Continente_Local', 'Continente_Visitante'], axis=1, inplace=True)

# ============================================================
# 2. LIMPIEZA
# ============================================================

print("Limpiando datos...")

df = df[
    ~df['Equipo_Local'].astype(str).str.contains('sub', case=False, na=False) &
    ~df['Equipo_Visitante'].astype(str).str.contains('sub', case=False, na=False)
].reset_index(drop=True)

columnas = [
    'Fecha', 'Equipo_Local', 'Equipo_Visitante', 'Resultado',
    'Goles_esperados_(xG)_Local', 'Goles_esperados_(xG)_Visitante',
    'Posesión_Local', 'Posesión_Visitante',
    'Remates_a_puerta_Local', 'Remates_a_puerta_Visitante',
    'Córneres_Local', 'Córneres_Visitante',
    'Pases_Local', 'Pases_Visitante',
    'Tarjetas_amarillas_Local', 'Tarjetas_amarillas_Visitante',
    'Faltas_Local', 'Faltas_Visitante',
    'Paradas_Local', 'Paradas_Visitante',
    'Puntos_Local', 'Puntos_Visitante',
    'Peso_Local', 'Peso_Visitante',
    'Valor_Mercado_Millones_Eur_Local', 'Valor_Mercado_Millones_Eur_Visitante',
]
df = df[[c for c in columnas if c in df.columns]]

df_mundial_raw = df[df['Fecha'] >= FECHA_CORTE].copy()
df_mundial_raw.to_csv(RUTA_OUT_PARTIDOS, index=False)

df = df[df['Fecha'] < FECHA_CORTE].copy()

df = df[
    df['Equipo_Local'].isin(equipos_mundial) |
    df['Equipo_Visitante'].isin(equipos_mundial)
]
df = df[df['Fecha'] >= FECHA_MINIMA].copy()

def limpiar_resultados_flashscore(texto):
    texto = str(texto).strip()
    m = re.match(r'^(\d+-\d+)\s*Finalizado', texto)
    if m:
        return pd.Series([m.group(1), None, None])
    m = re.match(r'^(\d+-\d+)\s*\(\s*(\d+-\d+)\s*\)\s*Tras la prórroga', texto)
    if m:
        return pd.Series([m.group(2), m.group(1), None])
    m = re.match(r'^(\d+-\d+)\s*\(\s*(\d+-\d+)\s*\)\s*Tras los penaltis', texto)
    if m:
        return pd.Series([m.group(2), m.group(2), m.group(1)])
    return pd.Series([texto, None, None])

df[['Resultado', 'Resultado_Prorroga', 'Resultado_Penaltis']] = \
    df['Resultado'].apply(limpiar_resultados_flashscore)

for col in ['Pases_Local', 'Pases_Visitante']:
    ext = df[col].astype(str).str.extract(r'(\d+)%\((\d+)/\d+\)')
    df[f'{col}_Pct']      = pd.to_numeric(ext[0], errors='coerce')
    df[f'{col}_Exitosos'] = pd.to_numeric(ext[1], errors='coerce')
df.drop(columns=['Pases_Local', 'Pases_Visitante'], inplace=True, errors='ignore')

for col in ['Posesión_Local', 'Posesión_Visitante',
            'Pases_Local_Pct', 'Pases_Visitante_Pct']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace('%', '').astype(float) / 100

df.dropna(subset=['Posesión_Local', 'Posesión_Visitante'], inplace=True)
df.reset_index(drop=True, inplace=True)

df_mundial_sorted = df_mundial_raw.sort_values('Fecha').reset_index(drop=True)
equipos_vistos, indices_debuts = set(), []
for idx, row in df_mundial_sorted.iterrows():
    if row['Equipo_Local'] not in equipos_vistos or row['Equipo_Visitante'] not in equipos_vistos:
        indices_debuts.append(idx)
        equipos_vistos.update([row['Equipo_Local'], row['Equipo_Visitante']])

if indices_debuts:
    df_mundial_first = df_mundial_sorted.iloc[indices_debuts].copy().reset_index(drop=True)
    df_mundial_first = df_mundial_first.dropna(axis=1, how='all')
    df = pd.concat([df, df_mundial_first], ignore_index=True)


# ============================================================
# 3. IMPUTACIÓN 
# ============================================================

print("Imputando valores faltantes...")
columnas_texto = [
    'Fecha', 'Equipo_Local', 'Equipo_Visitante',
    'Resultado', 'Resultado_Prorroga', 'Resultado_Penaltis',
]
columnas_numericas = [c for c in df.columns if c not in columnas_texto]

df_num_clean = df[columnas_numericas].dropna(axis=1, how='all')
columnas_numericas_reales = df_num_clean.columns.tolist()

imputador = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=42),
    max_iter=10, random_state=42, min_value=0,
)

datos_imputados = imputador.fit_transform(df_num_clean)

columnas_devueltas = imputador.get_feature_names_out(columnas_numericas_reales)

df_imp = pd.DataFrame(
    datos_imputados,
    columns=columnas_devueltas, 
    index=df.index,
)

df = pd.concat([df[columnas_texto], df_imp], axis=1)

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

print("Construyendo features...")

df[['Goles_Local', 'Goles_Visitante']] = \
    df['Resultado'].str.split('-', expand=True).apply(pd.to_numeric, errors='coerce')

df['Resultado_1X2'] = df.apply(
    lambda r: '1' if r['Goles_Local'] > r['Goles_Visitante']
    else ('X' if r['Goles_Local'] == r['Goles_Visitante'] else '2'), axis=1,
)
df['Ptos_Local']     = df.apply(
    lambda r: 3 if r['Goles_Local'] > r['Goles_Visitante']
    else (1 if r['Goles_Local'] == r['Goles_Visitante'] else 0), axis=1,
)
df['Ptos_Visitante'] = df.apply(
    lambda r: 3 if r['Goles_Visitante'] > r['Goles_Local']
    else (1 if r['Goles_Visitante'] == r['Goles_Local'] else 0), axis=1,
)

df['xGA_Local']     = df.get('Goles_esperados_(xG)_Visitante', np.nan)
df['xGA_Visitante'] = df.get('Goles_esperados_(xG)_Local', np.nan)

pares_metricas = [
    ('Goles_esperados_(xG)_Local', 'Goles_esperados_(xG)_Visitante', 'xG'),
    ('Remates_a_puerta_Local',      'Remates_a_puerta_Visitante',     'Remates_Puerta'),
    ('Córneres_Local',              'Córneres_Visitante',              'Corneres'),
    ('Paradas_Local',               'Paradas_Visitante',               'Paradas'),
    ('Pases_Local_Exitosos',        'Pases_Visitante_Exitosos',        'Pases_Exitosos'),
    ('Goles_Local',                 'Goles_Visitante',                 'Goles'),
    ('Puntos_Local',                'Puntos_Visitante',                'Puntos'),
    ('Valor_Mercado_Millones_Eur_Local', 'Valor_Mercado_Millones_Eur_Visitante', 'Valor_Mercado_Millones_Eur'),
]

for col_l, col_v, nombre in pares_metricas:
    if col_l in df.columns and col_v in df.columns:
        total = df[col_l] + df[col_v]
        df[f'{nombre}_Share_Local']     = np.where(total > 0, df[col_l] / total, 0.5)
        df[f'{nombre}_Share_Visitante'] = 1 - df[f'{nombre}_Share_Local']

# ─────────────────────────────────────────────────────────────
# 5. FORMATO LARGO Y MEDIAS MÓVILES
# ─────────────────────────────────────────────────────────────

print("Calculando medias móviles (ventanas 3 y 5)...")

df_local     = df.filter(like='_Local').copy()
df_visitante = df.filter(like='_Visitante').copy()
df_local.columns     = df_local.columns.str.replace('_Local', '')
df_visitante.columns = df_visitante.columns.str.replace('_Visitante', '')

for parte, tipo in [(df_local, 'Local'), (df_visitante, 'Visitante')]:
    parte['Tipo_Equipo']  = tipo
    parte['Fecha']        = df['Fecha'].values
    parte['Resultado_1X2'] = df['Resultado_1X2'].values

long_df = pd.concat([df_local, df_visitante], ignore_index=True)
long_df = long_df.sort_values('Fecha').reset_index(drop=True)

columnas_a_excluir = [
    'Fecha', 'Resultado_1X2', 'Tipo_Equipo', 'Puntos', 'Continente',
    'Peso', 'Valor_Mercado_Millones_Eur',
]

og_vars = [
    col for col in long_df.columns
    if col not in columnas_a_excluir
    and pd.api.types.is_numeric_dtype(long_df[col])
]

def add_team_stats(df_long, columnas_excluir, ventanas=None):
    if ventanas is None:
        ventanas = [3, 5]

    vars_to_avg = [
        col for col in df_long.columns
        if col not in columnas_excluir
        and pd.api.types.is_numeric_dtype(df_long[col])
    ] + ['Puntos']

    for var in vars_to_avg:
        df_long[f'avg_{var}_total'] = (
            df_long.groupby('Equipo')[var]
            .transform(lambda x: x.shift().expanding().mean())
        )

    for w in ventanas:
        for var in vars_to_avg:
            df_long[f'avg_{var}_{w}'] = (
                df_long.groupby('Equipo')[var]
                .transform(lambda x: x.shift().rolling(w, min_periods=1).mean())
            )

    xga_col = 'xGA'
    if xga_col in df_long.columns:
        for w in ventanas:
            df_long[f'avg_xGA_{w}'] = (
                df_long.groupby('Equipo')[xga_col]
                .transform(lambda x: x.shift().rolling(w, min_periods=1).mean())
            )
        df_long['avg_xGA_total'] = (
            df_long.groupby('Equipo')[xga_col]
            .transform(lambda x: x.shift().expanding().mean())
        )

        df_long['clean_sheet_rate_5'] = (
            df_long.groupby('Equipo')[xga_col]
            .transform(
                lambda x: x.shift()
                           .rolling(5, min_periods=1)
                           .apply(lambda v: (v < 0.5).mean(), raw=True)
            )
        )

    return df_long


long_df = add_team_stats(long_df, columnas_a_excluir, ventanas=VENTANAS)

xg_share_col = 'xG_Share'
if xg_share_col in long_df.columns:
    long_df['forma_vs_historia'] = (
        long_df['avg_xG_Share_3'] - long_df['avg_xG_Share_total']
    )

long_df.drop(columns=og_vars, inplace=True, errors='ignore')

print(f"long_df shape tras medias móviles: {long_df.shape}")

# ─────────────────────────────────────────────────────────────
# 6. JOIN DE MEDIAS AL DATASET DE PARTIDOS
# ─────────────────────────────────────────────────────────────

aux = df[['Fecha', 'Equipo_Local', 'Equipo_Visitante',
          'Resultado_1X2', 'Goles_Local', 'Goles_Visitante']].copy()

aux = aux.merge(
    long_df, left_on=['Fecha', 'Equipo_Local'],
    right_on=['Fecha', 'Equipo'], how='left',
)
df_final = aux.merge(
    long_df, left_on=['Fecha', 'Equipo_Visitante'],
    right_on=['Fecha', 'Equipo'], how='left',
)
df_final.drop(
    columns=['Equipo_x', 'Equipo_y', 'Resultado_1X2_x', 'Resultado_1X2_y',
             'Tipo_Equipo_x', 'Tipo_Equipo_y'],
    inplace=True, errors='ignore',
)
df_final.columns = df_final.columns.str.replace(r'_x$', '_Local',    regex=True)
df_final.columns = df_final.columns.str.replace(r'_y$', '_Visitante', regex=True)

# ─────────────────────────────────────────────────────────────
# 7. VARIABLES DIFERENCIA SOBRE LAS MEDIAS
# ─────────────────────────────────────────────────────────────

cols_avg_local = [
    c for c in df_final.columns
    if c.endswith(('_3_Local', '_5_Local', '_total_Local')) and c.startswith('avg_')
]
for col_l in cols_avg_local:
    col_v = col_l.replace('_Local', '_Visitante')
    if col_v in df_final.columns:
        nombre = col_l.replace('_Local', '')
        df_final[f'diff_{nombre}'] = df_final[col_l] - df_final[col_v]

for sufijo in ['trend_xG_5', 'clean_sheet_rate_5', 'forma_vs_historia']:
    col_l = f'{sufijo}_Local'
    col_v = f'{sufijo}_Visitante'
    if col_l in df_final.columns and col_v in df_final.columns:
        df_final[f'diff_{sufijo}'] = df_final[col_l] - df_final[col_v]

df_final['diff_Puntos']       = df_final['Puntos_Local']       - df_final['Puntos_Visitante']
df_final['Prob_Implicita_ELO'] = 1 / (1 + 10 ** (-df_final['diff_Puntos'] / 400))

def asignar_tier(puntos):
    if puntos >= 1700: return 1
    elif puntos >= 1600: return 2
    elif puntos >= 1500: return 3
    else: return 4

df_final['diff_Tier']          = (
    df_final['Puntos_Local'].apply(asignar_tier)
    - df_final['Puntos_Visitante'].apply(asignar_tier)
)
df_final['diff_Valor_Mercado'] = (
    df_final['Valor_Mercado_Millones_Eur_Local']
    - df_final['Valor_Mercado_Millones_Eur_Visitante']
)

# ─────────────────────────────────────────────────────────────
# 8. ELIMINACIÓN DE MULTICOLINEALIDAD
# ─────────────────────────────────────────────────────────────

exclude_corr = [
    'Fecha', 'Equipo_Local', 'Equipo_Visitante', 'Resultado', 'Resultado_Prorroga',
    'Resultado_Penaltis', 'Resultado_1X2', 'Tipo_Equipo',
    'Goles_Local', 'Goles_Visitante', 'Prob_Implicita_ELO',
    'Peso_Local', 'Peso_Visitante',
    'avg_Goles_esperados_(xG)_total_Local', 'avg_Goles_esperados_(xG)_total_Visitante',
    'diff_Valor_Mercado',
]
covariables  = [c for c in df_final.columns if c not in set(exclude_corr)]
corr_matrix  = df_final[covariables].corr().abs()
threshold    = 0.8
to_drop      = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] > threshold:
            col_i, col_j = corr_matrix.columns[i], corr_matrix.columns[j]
            var_i, var_j = df_final[col_i].var(), df_final[col_j].var()
            if 'diff' in col_i and 'diff' not in col_j:
                to_drop.add(col_j)
            elif 'diff' in col_j and 'diff' not in col_i:
                to_drop.add(col_i)
            else:
                to_drop.add(col_i if var_i < var_j else col_j)

protegidas = {
    'diff_xG_Share_3', 'diff_xG_Share_5', 'diff_xG_Share_total',
    'diff_Puntos', 'diff_Tier', 'diff_Valor_Mercado',
    'diff_trend_xG_5', 'diff_clean_sheet_rate_5', 'diff_forma_vs_historia',
}
to_drop = [c for c in to_drop if c not in set(exclude_corr) | protegidas]
df_reduced = df_final.drop(columns=to_drop)

print(f"Variables antes de eliminar correladas: {df_final.shape[1]}")
print(f"Variables tras eliminar correladas:     {df_reduced.shape[1]}")

# ─────────────────────────────────────────────────────────────
# 9. LIMPIEZA FINAL
# ─────────────────────────────────────────────────────────────

pattern_drop = r'avg_Peso_2|avg_Peso_3|avg_Peso_5|Puntos_Local|Puntos_Visitante|Valor_Mercado_Millones_Eur_'
df_reduced = df_reduced.loc[:, ~df_reduced.columns.str.contains(pattern_drop, regex=True)]
print(f"Shape final df_reduced: {df_reduced.shape}")

# ─────────────────────────────────────────────────────────────
# 10. EXPORTAR
# ─────────────────────────────────────────────────────────────

print("Exportando CSVs...")

aux_puntos = pd.concat([
    df[['Fecha', 'Equipo_Local',    'Puntos_Local',    'Valor_Mercado_Millones_Eur_Local']]
      .rename(columns={'Equipo_Local':    'Equipo',
                       'Puntos_Local':    'Puntos',
                       'Valor_Mercado_Millones_Eur_Local': 'Valor_Mercado_Millones_Eur'}),
    df[['Fecha', 'Equipo_Visitante', 'Puntos_Visitante', 'Valor_Mercado_Millones_Eur_Visitante']]
      .rename(columns={'Equipo_Visitante': 'Equipo',
                       'Puntos_Visitante': 'Puntos',
                       'Valor_Mercado_Millones_Eur_Visitante': 'Valor_Mercado_Millones_Eur'}),
], ignore_index=True)

aux_puntos = aux_puntos.sort_values('Fecha').drop_duplicates('Equipo', keep='last')

# EL ARREGLO ESTÁ AQUÍ: Eliminamos las columnas antiguas antes del merge para evitar duplicados
long_df.drop(columns=['Puntos', 'Valor_Mercado_Millones_Eur'], inplace=True, errors='ignore')

long_df = long_df.merge(
    aux_puntos[['Equipo', 'Puntos', 'Valor_Mercado_Millones_Eur']],
    on='Equipo', how='left',
)

df_mundial_export = long_df[long_df['Fecha'] >= FECHA_CORTE].copy()
df_historico_export = df_reduced[df_reduced['Fecha'] < FECHA_CORTE].copy()

df_historico_export.to_csv(RUTA_OUT_HISTORICO, index=False, encoding='utf-8-sig')
df_mundial_export.to_csv(RUTA_OUT_MUNDIAL,    index=False, encoding='utf-8-sig')

print(f"Datos historicos:, {df_historico_export.columns}")
print(f"Datos Mundial:, {df_mundial_export.columns}")

print(f"\n✓ datos_historicos.csv  → {df_historico_export.shape}")
print(f"✓ datos_mundial.csv     → {df_mundial_export.shape}")
print(f"✓ partidos_mundial.csv  → guardado antes del split")