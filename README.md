# Simulaciones Mundial 2026 ⚽🏆

¿Quién va a ganar el Mundial 2026? Este repositorio responde con datos: scrapea todo el histórico de estadísticas de las selecciones, entrena dos modelos XGBoost y simula **50.000 mundiales** con Monte Carlo para estimar el rendimiento esperado de cada selección — hasta predecir los 104 partidos del torneo. Iré actualizando y explicando todo más en detalle en `@jyts__`.

---

## 🔄 El pipeline, paso a paso

Todo el código del proyecto centralizado se encuentra dentro del directorio [`code/`](code/).

### 1️⃣ Scraping — [`code/Scrapper.py`](code/Scrapper.py) y [`code/Scrapper_nuevos.py`](code/Scrapper_nuevos.py)
Los scrapers (Selenium + BeautifulSoup) se descargan **todo el histórico de estadísticas de FlashScore**: resultados, xG, posesión, remates a puerta, córneres, faltas, paradas, pases... de los últimos partidos de cada selección, más el ranking FIFA. Si no quieres scrapear desde cero, los datos ya descargados están en la raíz dentro de [`Data/`](Data/).

### 2️⃣ Limpieza de datos — [`code/Data_Cleaning.py`](code/Data_Cleaning.py)
Este script hace el JOIN de las distintas fuentes descargadas y realiza una limpieza exhaustiva para **pasar del HTML scrapeado en bruto a un dataset modelable**: una fila por partido con las estadísticas de los dos equipos.

### 3️⃣ Ingeniería de variables y Modelado — [`code/Modelling.ipynb`](code/Modelling.ipynb)
En este notebook se procesan las variables y se entrenan los modelos:
* **Ingeniería de variables**: Sobre los datos limpios se construyen **medias móviles (últimos 5 partidos e histórico), ratios y diferencias** entre equipos para capturar el *estado de forma* de cada selección justo antes del partido (puntos Elo/FIFA, probabilidad implícita, tiers, pesos por confederación).
* **Modelos XGBoost**: 
  1. **Modelo de goles**: dos regresores XGBoost con objetivo Tweedie (ideal para fútbol) que predicen los goles esperados de cada equipo.
  2. **Modelo de resultado**: un clasificador XGBoost 1X2 que usa como meta-variables las predicciones de goles del primero, con **probabilidades calibradas** (calibración isotónica).

### 4️⃣ El comportamiento estocástico: Monte Carlo con 10.000 mundiales
¿Por qué no basta con simular un mundial? Porque la realidad es **estocástica**. Para capturar el efecto de la varianza, simulamos **10.000 mundiales diferentes** a través de las probabilidades estimadas por el modelo en todos los cruces del torneo — fase de grupos partido a partido, mejores terceros, dieciseisavos, octavos, cuartos, semifinales y final.

### 5️⃣ Predicción de los partidos — [`code/prediccion_mundial.py`](code/prediccion_mundial.py) y [`code/prediccion_fasefinal.py`](code/prediccion_fasefinal.py)
Scripts dedicados a ejecutar el pipeline end-to-end para generar las predicciones de la fase de grupos, el desarrollo completo del cuadro definitivo y el Monte Carlo general. 

**Resultados consolidados** en [`Predicciones/PREDICCIONES.md`](Predicciones/PREDICCIONES.md): marcador más probable y probabilidades 1X2 de los 104 partidos, clasificaciones de los 12 grupos, cuadro hasta la final y probabilidades de campeón por selección.

### 🌐 Web de predicciones diarias — [`docs/`](docs/)
Una página estática (lista para **GitHub Pages**) alojada en la carpeta `docs/` con las predicciones de cada día: probabilidades calibradas, córners, tarjetas y el cuadro interactivo. Un workflow de GitHub Actions ([`.github/workflows/actualizar_web.yml`](.github/workflows/actualizar_web.yml)) la regenera a diario automáticamente.

Para activarla: en GitHub ve a **Settings → Pages → Source: Deploy from a branch → `main` / `docs/`**. La web quedará en `https://<usuario>.github.io/Simulaciones_Mundial/`.

---

## 📂 Estructura del repositorio

```
├── code/                     # Scripts y notebooks de todo el pipeline
│   ├── Data_Cleaning.py      # JOIN y limpieza → dataset modelable
│   ├── Modelling.ipynb       # Ingeniería de variables, XGBoost y Monte Carlo
│   ├── Scrapper.py           # Extracción de datos históricos de FlashScore (Selenium)
│   ├── Scrapper_nuevos.py    # Extracción de nuevos datos/actualizaciones
│   ├── prediccion_fasefinal.py # Script de predicción para las eliminatorias directas
│   └── prediccion_mundial.py   # Pipeline de predicción completo del torneo
├── Data/                     # Datos scrapeados y procesados (CSV)
├── docs/                     # Web estática desplegada en GitHub Pages (index.html)
└── Predicciones/             # Predicción de los 104 partidos + Monte Carlo
└── PREDICCIONES.md
```

## 🚀 Replicarlo

```bash
pip install pandas numpy xgboost scikit-learn joblib tqdm

# (Opcional) actualizar u obtener los datos desde cero
python code/Scrapper.py
python code/Scrapper_nuevos.py

# Ejecutar limpieza de datos
python code/Data_Cleaning.py

# El entrenamiento y calibración se pueden revisar en el notebook: code/Modelling.ipynb

# Predicción completa del Mundial (entrena + simula 10.000 mundiales)
python code/prediccion_mundial.py
python code/prediccion_fasefinal.py
```

## 💡 Inspírate

Aprovecha este proyecto para hacer algo aún más profundo a partir de él: incorporar más datos, probar otros modelos u otros enfoques para la modelización y la simulación. Estaré encantado de que me cuentes lo que has construido en `@jyts__`.
