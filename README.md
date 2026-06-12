# Simulaciones Mundial 2026
Código para hacer el web scrapping de las estadísticas de los últimos partidos de selecciones, ajustar un modelo XGBoost para predecir los resultados y una simulación de Montecarlo de 10.000 mundiales simulados para predecir quién ganará y el performance esperado de cada selección. A mayores se incluyen todos los datos scrappeados. Iré actualizando y explicando todo más en detalle en `@jyts__`. 

# 🏆 Predicción de los 104 partidos del Mundial 2026
En la carpeta [`Predicciones/`](Predicciones/) está la **predicción completa del torneo** generada con el modelo de este repositorio (ejecutando `prediccion_mundial.py`, un port a script de los notebooks):

- [`PREDICCIONES.md`](Predicciones/PREDICCIONES.md) — informe con los 104 partidos: 72 de fase de grupos (marcador más probable + probabilidades 1X2), clasificaciones predichas, cuadro completo de eliminatorias hasta la final y probabilidades de campeón por selección.
- `predicciones_fase_grupos.csv`, `clasificacion_grupos.csv`, `predicciones_eliminatorias.csv`, `probabilidades_montecarlo.csv` — los mismos resultados en formato datos.

Para regenerarlas: `python prediccion_mundial.py && python generar_informe.py` (entrena los modelos si no existen los `.pkl` y corre 10.000 simulaciones de Monte Carlo).

# Instrucciones de Uso
## 1. Extracción de Datos (Scraping)
Si deseas actualizar el histórico de partidos o replicar la captura de datos desde cero, ejecuta el script `Scrapper.py`.

Si no, tienes disponibles los datos descargados en la carpeta `/Data`.

## 2. Limpieza de Datos
El Notebook `Data Cleaning.ipynb` se encarga de hacer el JOIN de los distintos datos descragados y de la limpieza y construcción de variables para la posterior modelización. 

## 3. Ejecución del Modelo y Simulación de Momte Carlo
Tienes el Notebook `Modelling.ipynb` para ejecutar la ingeniería de variables, el entrenamiento de los modelos y correr las simulaciones de Montecarlo. 

## 4. Inspírate!
Aprovecha este proyecto para hacer algo aún más profundo a partir de el. Hay muchas mejoras posibles, desde incorporar más datos a considerar otros modelos u otros approaches para toda la modelización y simulación. 

Estaré encantado de que me cuentes lo que has construido en `@jyts__`
