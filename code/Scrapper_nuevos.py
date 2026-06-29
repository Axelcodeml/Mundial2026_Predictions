from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import requests
import time
import random
import pandas as pd
import re
import os

# ==========================================
# FUNCIONES DE EXTRACCIÓN Y PROCESAMIENTO
# ==========================================

def extraer_partido_completo(url, driver):
    datos_partido = {'fecha': 'Desconocida', 'url': url, 'equipo_local': 'Desconocido', 'equipo_visitante': 'Desconocido', 'resultado': 'Desconocido', 'estadisticas': {}}
    try:
        driver.get(url)
        time.sleep(5) 
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        fecha_element = soup.find(class_=lambda x: x and isinstance(x, str) and 'startTime' in x)
        if fecha_element: datos_partido['fecha'] = fecha_element.text.strip()
            
        enlaces_equipos = soup.find_all('a', class_=lambda x: x and isinstance(x, str) and 'participantName' in x)
        equipos_unicos = []
        for enlace in enlaces_equipos:
            nombre = enlace.text.strip()
            if nombre and nombre not in equipos_unicos: equipos_unicos.append(nombre)
        if len(equipos_unicos) >= 2:
            datos_partido['equipo_local'] = equipos_unicos[0]
            datos_partido['equipo_visitante'] = equipos_unicos[1]
            
        marcador = soup.find(class_=lambda x: x and isinstance(x, str) and 'detailScore' in x)
        if marcador: datos_partido['resultado'] = marcador.text.replace('\n', '').strip()

        todos_los_divs = soup.find_all('div')
        for div in todos_los_divs:
            hijos = div.find_all('div', recursive=False)
            if len(hijos) == 3:
                valor_local, categoria, valor_visitante = hijos[0].text.strip(), hijos[1].text.strip(), hijos[2].text.strip()
                if valor_local and categoria and valor_visitante and any(c.isalpha() for c in categoria) and len(categoria) < 30:
                    datos_partido['estadisticas'][categoria] = {'local': valor_local, 'visitante': valor_visitante}
    except Exception as e:
        print(f"Error en {url}: {e}")
    return datos_partido

def aplanar_datos(datos_partido):
    datos_planos = {'Fecha': datos_partido['fecha'], 'URL': datos_partido['url'], 'Equipo_Local': datos_partido['equipo_local'], 'Equipo_Visitante': datos_partido['equipo_visitante'], 'Resultado': datos_partido['resultado']}
    for categoria, valores in datos_partido['estadisticas'].items():
        nombre_col = categoria.replace(" ", "_")
        datos_planos[f"{nombre_col}_Local"] = valores['local']
        datos_planos[f"{nombre_col}_Visitante"] = valores['visitante']
    return datos_planos


# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    carpeta_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RUTA_URLS, RUTA_CSV_PARTIDOS = os.path.join(carpeta_raiz, "Data", "nuevos_urls.txt"), os.path.join(carpeta_raiz, "Data", "partidos.csv")
    
    opciones = Options()
    # opciones.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opciones)

    # --- FASE 1: SCRAPING Y ACTUALIZACIÓN ---
    with open(RUTA_URLS, 'r', encoding='utf-8') as f:
        lista_urls = [l.strip() for l in f if l.strip()]

    df_existente = pd.read_csv(RUTA_CSV_PARTIDOS) if os.path.exists(RUTA_CSV_PARTIDOS) else pd.DataFrame()
    nuevos_datos = []

    for url in lista_urls:
        print(f"Procesando: {url}")
        datos = aplanar_datos(extraer_partido_completo(url, driver))
        if datos['Equipo_Local'] != 'Desconocido': nuevos_datos.append(datos)
        time.sleep(random.uniform(3, 5))

    # Combinar, eliminar duplicados (se queda el último registro que es el más reciente)
    df_final = pd.concat([df_existente, pd.DataFrame(nuevos_datos)], ignore_index=True)
    df_final.drop_duplicates(subset=['URL'], keep='last', inplace=True)
    df_final.to_csv(RUTA_CSV_PARTIDOS, index=False, encoding='utf-8-sig')
    
    print(f"Proceso finalizado. Total registros únicos: {len(df_final)}")