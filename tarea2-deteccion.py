# tarea2-deteccion.py
# CC5213 - TAREA 2 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 20 septiembre de 2024
# Alumno: [Juan Vicente Onetto]



import sys
import os
from util import escribir_lista_de_columnas_en_archivo
from collections import defaultdict, Counter
from tqdm import tqdm

def cargar_ventanas_similares(archivo_ventanas_similares):
    ventanas = []
    with open(archivo_ventanas_similares, 'r') as f:
        for linea in f:
            partes = linea.strip().split('\t')
            if len(partes) != 4:
                continue
            ventana = {
                'archivo_Q': partes[0],
                'inicio_Q': float(partes[1]),
                'archivo_R': partes[2],
                'inicio_R': float(partes[3])
            }
            ventanas.append(ventana)
    return ventanas

def tarea2_deteccion(archivo_ventanas_similares, archivo_detecciones, ventana_duracion=5.0, k_min=2, margen_desfase=1.0, umbral_confianza=1):
    """
    Detecta las canciones emitidas en los audios de Q basándose en las ventanas similares encontradas.
    
    Parámetros:
    - archivo_ventanas_similares: Archivo de entrada con las ventanas similares.
    - archivo_detecciones: Archivo de salida con las detecciones.
    - ventana_duracion: Duración de cada ventana en segundos.
    - k_min: Número mínimo de ventanas similares para considerar una detección.
    - margen_desfase: Margen de tolerancia para el desfase de tiempo (en segundos).
    """
    # Cargar ventanas similares
    print("Cargando ventanas similares...")
    ventanas = cargar_ventanas_similares(archivo_ventanas_similares)
    
    # Agrupar por archivo de Q
    print("Agrupando ventanas por archivo de Q...")
    ventanas_por_Q = defaultdict(list)
    for v in ventanas:
        ventanas_por_Q[v['archivo_Q']].append(v)
    
    detecciones = []
    
    # Procesar cada archivo de Q
    print("Procesando detecciones...")
    for archivo_Q, ventanas_Q in tqdm(ventanas_por_Q.items(), desc="Archivos Q"):
        ventanas_por_R = defaultdict(list)
        for v in ventanas_Q:
            ventanas_por_R[v['archivo_R']].append(v)
        
        for archivo_R, ventanas_R in ventanas_por_R.items():
            ventanas_R_sorted = sorted(ventanas_R, key=lambda x: x['inicio_Q'])
            
            tiempos_Q = [v['inicio_Q'] for v in ventanas_R_sorted]
            tiempos_R = [v['inicio_R'] for v in ventanas_R_sorted]
            
            desfases = [q - r for q, r in zip(tiempos_Q, tiempos_R)]
            
            desfases_redondeados = [round(d, 1) for d in desfases]
            contador = Counter(desfases_redondeados)
            desfase_mas_comun, cuenta = contador.most_common(1)[0]
            
            ventanas_filtradas = [v for v, d in zip(ventanas_R_sorted, desfases_redondeados) if abs(d - desfase_mas_comun) <= margen_desfase]
            
            if len(ventanas_filtradas) < k_min:
                continue
            
            inicio_Q = ventanas_filtradas[0]['inicio_Q']
            fin_Q = ventanas_filtradas[-1]['inicio_Q'] + ventana_duracion  
            largo = fin_Q - inicio_Q
            
            confianza = len(ventanas_filtradas)  
            if confianza < umbral_confianza:
                continue
            
            detecciones.append([
                archivo_Q,
                inicio_Q,
                largo,
                archivo_R,
                confianza
            ])
    
    print(f"Escribiendo detecciones en {archivo_detecciones}...")
    escribir_lista_de_columnas_en_archivo(detecciones, archivo_detecciones)
    print("Detección completada.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python tarea2-deteccion.py [archivo_ventanas_similares] [archivo_detecciones]")
        sys.exit(1)
    
    archivo_ventanas_similares = sys.argv[1]
    archivo_detecciones = sys.argv[2]
    
    tarea2_deteccion(archivo_ventanas_similares, archivo_detecciones, ventana_duracion=5.0, k_min=2, margen_desfase=1.0, umbral_confianza=1)
