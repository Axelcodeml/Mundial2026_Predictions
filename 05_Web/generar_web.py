# -*- coding: utf-8 -*-
"""Genera la web estática de predicciones (docs/index.html, lista para GitHub Pages).

Para cada partido de la fase de grupos calcula con el modelo del repo:
  - Probabilidades 1X2 calibradas (T=1, sin afilar: las honestas)
  - Marcador más probable y goles esperados (xG del modelo)
  - Córners esperados y P(más de 7.5 / 8.5 / 9.5)  [Poisson sobre medias por equipo]
  - Tarjetas amarillas esperadas y P(más de 3.5 / 4.5)

La sección "Partidos de hoy" se resuelve en el navegador con la fecha local del
visitante, así la página se actualiza sola cada día. Un workflow de GitHub
Actions la regenera a diario por si cambian los datos o el modelo.

Uso:  python 05_Web/generar_web.py   (requiere los .pkl entrenados; si no
      existen, ejecuta antes 04_Prediccion/prediccion_mundial.py)
"""

import os
import sys
import json
import math
from datetime import datetime, timezone

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RAIZ)
sys.path.insert(0, os.path.join(RAIZ, '04_Prediccion'))

import prediccion_mundial as pm  # noqa: E402  (reutilizamos el pipeline del repo)


def p_poisson_mas_de(lam, linea):
    """P(N >= ceil(linea)) con N ~ Poisson(lam)."""
    k_min = int(linea) + 1
    return 1 - sum(math.exp(-lam) * lam ** k / math.factorial(k) for k in range(k_min))


def construir_datos():
    df_mundial_grupos, df_vars, grupos, fechas_reales = pm.cargar_mundial()

    # Probabilidades calibradas sin temperatura: las adecuadas para mostrar
    pred = pm.pipeline_prediccion(df_mundial_grupos, sede_neutral=True, T=1.0)
    pred['Grupo'] = df_mundial_grupos['Grupo'].values

    stats = df_vars.set_index('Equipo')
    mapa_fechas = {(r['Equipo_Local'], r['Equipo_Visitante']): r['Fecha']
                   for _, r in fechas_reales.iterrows()}

    partidos = []
    for _, r in pred.iterrows():
        a, b = r['Equipo_Local'], r['Equipo_Visitante']
        p1, px, p2 = r['Prob_Local'], r['Prob_Empate'], r['Prob_Visitante']
        s = p1 + px + p2
        p1, px, p2 = p1 / s, px / s, p2 / s
        res = ['1', 'X', '2'][int(max(range(3), key=[p1, px, p2].__getitem__))]
        gl, gv = pm.marcador_mas_probable(r['xG_Modelo_Local'], r['xG_Modelo_Visitante'], res)

        # Córners y tarjetas: promedio de la estimación reciente (últimos 5) e histórica
        lam_cor = (stats.loc[a, 'avg_Córneres_5'] + stats.loc[b, 'avg_Córneres_5'] +
                   stats.loc[a, 'avg_Córneres_total'] + stats.loc[b, 'avg_Córneres_total']) / 2
        lam_tar = (stats.loc[a, 'avg_Tarjetas_amarillas_5'] + stats.loc[b, 'avg_Tarjetas_amarillas_5'] +
                   stats.loc[a, 'avg_Tarjetas_amarillas_total'] + stats.loc[b, 'avg_Tarjetas_amarillas_total']) / 2

        partidos.append({
            'fecha': mapa_fechas.get((a, b), ''),
            'grupo': r['Grupo'], 'local': a, 'visitante': b,
            'marcador': f'{gl}-{gv}',
            'p1': round(p1 * 100, 1), 'px': round(px * 100, 1), 'p2': round(p2 * 100, 1),
            'xgl': round(float(r['xG_Modelo_Local']), 2), 'xgv': round(float(r['xG_Modelo_Visitante']), 2),
            'cor': round(lam_cor, 1),
            'c75': round(p_poisson_mas_de(lam_cor, 7.5) * 100), 'c85': round(p_poisson_mas_de(lam_cor, 8.5) * 100),
            'c95': round(p_poisson_mas_de(lam_cor, 9.5) * 100),
            'tar': round(lam_tar, 1),
            't35': round(p_poisson_mas_de(lam_tar, 3.5) * 100), 't45': round(p_poisson_mas_de(lam_tar, 4.5) * 100),
        })
    partidos.sort(key=lambda p: (p['fecha'], p['grupo']))

    mc = pd.read_csv('Predicciones/probabilidades_montecarlo.csv', index_col=0)
    campeon = [{'equipo': eq, 'campeon': float(r['Campeon']), 'final': float(r['Final']),
                'semis': float(r['Semis'])} for eq, r in mc.head(12).iterrows()]

    elim = pd.read_csv('Predicciones/predicciones_eliminatorias.csv')
    cuadro = [{'fase': r['Fase'], 'fechas': r['Fechas'], 'local': r['Local'], 'visitante': r['Visitante'],
               'marcador': r['Marcador_Predicho'], 'avanza': r['Avanza'],
               'p1': float(r['Prob_1']), 'px': float(r['Prob_X']), 'p2': float(r['Prob_2'])}
              for _, r in elim.iterrows()]

    return partidos, campeon, cuadro


PLANTILLA = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mundial 2026 · Predicciones diarias del modelo</title>
<meta name="description" content="Probabilidades de ganar, córners, tarjetas y marcador predicho para cada partido del Mundial 2026, generadas con XGBoost y simulación de Monte Carlo.">
<style>
:root{--bg:#0d1220;--panel:#161d31;--panel2:#1c2540;--tx:#eef1f8;--tx2:#9aa5c0;--lin:#2a3554;
--v:#34d399;--e:#fbbf24;--d:#fb7185;--ac:#60a5fa}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;padding-bottom:48px}
.wrap{max-width:980px;margin:0 auto;padding:0 16px}
header{padding:34px 0 10px;text-align:center}
header h1{font-size:1.7rem;font-weight:700}
header p{color:var(--tx2);font-size:.95rem;max-width:640px;margin:8px auto 0}
.badge{display:inline-block;background:var(--panel2);border:1px solid var(--lin);color:var(--tx2);
border-radius:999px;padding:2px 12px;font-size:.8rem;margin-top:10px}
h2{font-size:1.15rem;margin:34px 0 14px;display:flex;align-items:center;gap:8px}
h2 small{color:var(--tx2);font-weight:400;font-size:.85rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--lin);border-radius:14px;padding:16px}
.card.hoy{border-color:var(--ac)}
.enc{display:flex;justify-content:space-between;align-items:baseline;font-size:.8rem;color:var(--tx2);margin-bottom:8px}
.eqs{display:flex;justify-content:space-between;align-items:center;gap:8px;font-weight:600;font-size:1.02rem}
.marc{background:var(--panel2);border:1px solid var(--lin);border-radius:10px;padding:2px 10px;font-size:1.05rem;white-space:nowrap}
.barra{display:flex;height:9px;border-radius:6px;overflow:hidden;margin:12px 0 4px;background:var(--panel2)}
.barra i{display:block;height:100%}
.leyenda{display:flex;justify-content:space-between;font-size:.78rem;color:var(--tx2)}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px;font-size:.8rem}
.stat{background:var(--panel2);border-radius:10px;padding:8px 10px}
.stat b{display:block;font-size:.86rem;color:var(--tx)}
.stat span{color:var(--tx2)}
.pasado{opacity:.55}
.etiq{font-size:.72rem;border:1px solid var(--lin);border-radius:6px;padding:1px 6px;color:var(--tx2)}
table{width:100%;border-collapse:collapse;font-size:.85rem}
th,td{padding:7px 8px;text-align:left;border-bottom:1px solid var(--lin)}
th{color:var(--tx2);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.vd{color:var(--v)}.em{color:var(--e)}.dr{color:var(--d)}
details{background:var(--panel);border:1px solid var(--lin);border-radius:12px;margin-bottom:10px;overflow:hidden}
summary{cursor:pointer;padding:12px 16px;font-weight:600;list-style:none;display:flex;justify-content:space-between}
summary::after{content:"+";color:var(--tx2)}
details[open] summary::after{content:"–"}
details .inner{padding:0 16px 14px;overflow-x:auto}
.mcbar{height:8px;background:var(--panel2);border-radius:5px;overflow:hidden}
.mcbar i{display:block;height:100%;background:var(--ac)}
.aviso{background:var(--panel);border:1px solid var(--lin);border-left:4px solid var(--e);border-radius:0 12px 12px 0;
padding:14px 16px;font-size:.85rem;color:var(--tx2);margin-top:36px}
footer{text-align:center;color:var(--tx2);font-size:.8rem;margin-top:30px}
footer a{color:var(--ac);text-decoration:none}
.vacio{background:var(--panel);border:1px dashed var(--lin);border-radius:12px;padding:22px;text-align:center;color:var(--tx2)}
@media(max-width:560px){.stats{grid-template-columns:1fr}}
</style>
</head>
<body>
<header class="wrap">
  <h1>⚽ Mundial 2026 — predicciones del modelo</h1>
  <p>Probabilidades de victoria, córners, tarjetas y marcador más probable para cada partido,
  generadas con dos modelos XGBoost (goles y resultado 1X2 calibrado) y 10.000 mundiales simulados por Monte Carlo.</p>
  <span class="badge">Datos regenerados: __FECHA_GEN__ · la sección «Hoy» se actualiza sola con tu fecha local</span>
</header>

<main class="wrap">
  <h2 id="t-hoy">📅 Partidos de hoy</h2>
  <div id="hoy" class="cards"></div>

  <h2 id="t-man">🌙 Mañana <small>(los horarios de EE. UU. pueden caer de madrugada en hora europea)</small></h2>
  <div id="manana" class="cards"></div>

  <h2>🏆 Probabilidades de campeón <small>Monte Carlo · 10.000 simulaciones</small></h2>
  <div class="card"><div class="inner" style="overflow-x:auto">
  <table id="tabla-mc"><thead><tr><th>#</th><th>Selección</th><th class="num">Campeón</th><th class="num">Final</th><th class="num">Semis</th><th style="width:34%"></th></tr></thead><tbody></tbody></table>
  </div></div>

  <h2>📋 Fase de grupos completa <small>72 partidos · probabilidades calibradas</small></h2>
  <div id="grupos"></div>

  <h2>🗺️ Cuadro de eliminatorias predicho <small>resultado más probable de cada cruce</small></h2>
  <div id="cuadro"></div>

  <div class="aviso"><b>⚠️ Esto son probabilidades, no certezas.</b> El modelo acierta ~65% de los ganadores en datos
  de prueba; un favorito del 80% pierde 1 de cada 5 veces. Las estimaciones de córners y tarjetas son aproximaciones
  Poisson sobre las medias de cada selección. Nada de esta página es consejo de apuestas: si juegas, que sea solo
  entretenimiento, con límites y con dinero que no te duela perder.</div>

  <footer>Generado automáticamente con el modelo de
  <a href="https://github.com/jytsss/Simulaciones_Mundial">Simulaciones_Mundial</a> · @jyts__</footer>
</main>

<script>
const PARTIDOS = __PARTIDOS_JSON__;
const CAMPEON = __CAMPEON_JSON__;
const CUADRO = __CUADRO_JSON__;

const isoLocal = d => d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0');
const hoy = new Date();
const HOY = isoLocal(hoy);
const MAN = isoLocal(new Date(hoy.getTime()+86400000));
const FIN_GRUPOS = PARTIDOS.length ? PARTIDOS[PARTIDOS.length-1].fecha : '2026-06-28';

function fmtFecha(f){
  const [y,m,d] = f.split('-');
  const meses=['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic'];
  return d+' '+meses[+m-1];
}

function tarjetaPartido(p, esHoy){
  const div = document.createElement('div');
  div.className = 'card' + (esHoy ? ' hoy' : '');
  div.innerHTML = `
    <div class="enc"><span>Grupo ${p.grupo} · ${fmtFecha(p.fecha)}</span><span class="etiq">xG ${p.xgl} – ${p.xgv}</span></div>
    <div class="eqs"><span>${p.local}</span><span class="marc">${p.marcador}</span><span style="text-align:right">${p.visitante}</span></div>
    <div class="barra">
      <i style="width:${p.p1}%;background:var(--v)"></i>
      <i style="width:${p.px}%;background:var(--e)"></i>
      <i style="width:${p.p2}%;background:var(--d)"></i>
    </div>
    <div class="leyenda"><span class="vd">1 · ${p.p1}%</span><span class="em">X · ${p.px}%</span><span class="dr">2 · ${p.p2}%</span></div>
    <div class="stats">
      <div class="stat"><b>⛳ Córners: ${p.cor} esperados</b>
        <span>+7.5: ${p.c75}% · +8.5: ${p.c85}% · +9.5: ${p.c95}%</span></div>
      <div class="stat"><b>🟨 Tarjetas: ${p.tar} esperadas</b>
        <span>+3.5: ${p.t35}% · +4.5: ${p.t45}%</span></div>
    </div>`;
  return div;
}

function pintarDia(idCont, fecha, esHoy){
  const cont = document.getElementById(idCont);
  const lista = PARTIDOS.filter(p => p.fecha === fecha);
  if (!lista.length){
    const v = document.createElement('div');
    v.className = 'vacio';
    v.style.gridColumn = '1/-1';
    v.textContent = (fecha > FIN_GRUPOS)
      ? 'Fase de grupos terminada: consulta el cuadro de eliminatorias predicho más abajo. Los cruces reales se conocerán al cerrar los grupos.'
      : 'No hay partidos programados para este día.';
    cont.appendChild(v);
    return;
  }
  lista.forEach(p => cont.appendChild(tarjetaPartido(p, esHoy)));
}
pintarDia('hoy', HOY, true);
pintarDia('manana', MAN, false);

const tb = document.querySelector('#tabla-mc tbody');
CAMPEON.forEach((c,i) => {
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${i+1}</td><td>${c.equipo}</td><td class="num"><b>${c.campeon.toFixed(1)}%</b></td>
  <td class="num">${c.final.toFixed(1)}%</td><td class="num">${c.semis.toFixed(1)}%</td>
  <td><div class="mcbar"><i style="width:${Math.min(100, c.campeon*3.3)}%"></i></div></td>`;
  tb.appendChild(tr);
});

const contG = document.getElementById('grupos');
[...new Set(PARTIDOS.map(p=>p.grupo))].sort().forEach(g => {
  const det = document.createElement('details');
  const filas = PARTIDOS.filter(p=>p.grupo===g).map(p => `
    <tr class="${p.fecha < HOY ? 'pasado' : ''}">
      <td>${fmtFecha(p.fecha)}</td><td>${p.local} – ${p.visitante}</td><td class="num"><b>${p.marcador}</b></td>
      <td class="num vd">${p.p1}%</td><td class="num em">${p.px}%</td><td class="num dr">${p.p2}%</td>
      <td class="num">${p.cor} <span style="color:var(--tx2)">(+7.5: ${p.c75}%)</span></td>
      <td class="num">${p.tar} <span style="color:var(--tx2)">(+3.5: ${p.t35}%)</span></td>
    </tr>`).join('');
  det.innerHTML = `<summary>Grupo ${g}</summary><div class="inner">
    <table><thead><tr><th>Fecha</th><th>Partido</th><th class="num">Pred.</th><th class="num">1</th>
    <th class="num">X</th><th class="num">2</th><th class="num">Córners</th><th class="num">Tarjetas</th></tr></thead>
    <tbody>${filas}</tbody></table></div>`;
  contG.appendChild(det);
});

const contC = document.getElementById('cuadro');
[...new Set(CUADRO.map(c=>c.fase))].forEach(f => {
  const det = document.createElement('details');
  if (f === 'Final') det.open = true;
  const filas = CUADRO.filter(c=>c.fase===f).map(c => `
    <tr><td>${c.local} – ${c.visitante}</td><td class="num"><b>${c.marcador}</b></td><td><b>${c.avanza}</b></td>
    <td class="num vd">${c.p1}%</td><td class="num em">${c.px}%</td><td class="num dr">${c.p2}%</td></tr>`).join('');
  det.innerHTML = `<summary>${f} <span class="etiq" style="margin-left:auto;margin-right:10px">${CUADRO.find(c=>c.fase===f).fechas}</span></summary>
    <div class="inner"><table><thead><tr><th>Cruce</th><th class="num">Pred.</th><th>Avanza</th>
    <th class="num">1</th><th class="num">X</th><th class="num">2</th></tr></thead><tbody>${filas}</tbody></table></div>`;
  contC.appendChild(det);
});
</script>
</body>
</html>
"""


if __name__ == '__main__':
    partidos, campeon, cuadro = construir_datos()
    html = (PLANTILLA
            .replace('__PARTIDOS_JSON__', json.dumps(partidos, ensure_ascii=False))
            .replace('__CAMPEON_JSON__', json.dumps(campeon, ensure_ascii=False))
            .replace('__CUADRO_JSON__', json.dumps(cuadro, ensure_ascii=False))
            .replace('__FECHA_GEN__', datetime.now(timezone.utc).strftime('%d-%m-%Y %H:%M UTC')))
    os.makedirs('docs', exist_ok=True)
    with open('docs/index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Web generada en docs/index.html ({len(partidos)} partidos, '
          f'{len(cuadro)} cruces, {len(campeon)} equipos en la tabla MC).')
