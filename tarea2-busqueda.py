# tarea2-busqueda.py

# CC5213 - TAREA 2 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 20 septiembre de 2024
# Alumno: [Juan Vicente Onetto]

import sys
import os
import numpy as np
from util import listar_archivos_con_extension, leer_objeto, escribir_lista_de_columnas_en_archivo
from pyflann import FLANN
from tqdm import tqdm

def cargar_descriptores(carpeta_descriptores):
    descriptores = []
    archivos = listar_archivos_con_extension(carpeta_descriptores, ".pkl")
    for archivo in archivos:
        ruta_archivo = os.path.join(carpeta_descriptores, archivo)
        lista_descriptores = leer_objeto(carpeta_descriptores, archivo)
        descriptores.extend(lista_descriptores)
    return descriptores

def tarea2_busqueda(carpeta_descriptores_Q, carpeta_descriptores_R, archivo_ventanas_similares, k):
    """
    Realiza la búsqueda de las ventanas más similares de R para cada ventana en Q.
    """
    # Cargar descriptores
    print("Cargando descriptores de Q...")
    descriptores_Q = cargar_descriptores(carpeta_descriptores_Q)
    print(f"Total de descriptores en Q: {len(descriptores_Q)}")

    print("Cargando descriptores de R...")
    descriptores_R = cargar_descriptores(carpeta_descriptores_R)
    print(f"Total de descriptores en R: {len(descriptores_R)}")

    if len(descriptores_R) == 0:
        print("No se encontraron descriptores en R.")
        sys.exit(1)

    # Preparar datos para FLANN
    print("Preparando datos para FLANN...")
    # Extraer los vectores de características de R
    data_R = np.array([d['descriptor'] for d in descriptores_R], dtype=np.float32)

    print("Creando índice FLANN...")
    flann = FLANN()
    flann_params = flann.build_index(data_R, algorithm='kdtree', trees=5)

    # Realizar la búsqueda
    print("Realizando búsqueda de vecinos más cercanos...")
    print("Usando k =" + str(k))
    resultados = []
    for q in tqdm(descriptores_Q, desc="Procesando Q"):
        descriptor_q = np.array(q['descriptor'], dtype=np.float32).reshape(1, -1)
        indices, _ = flann.nn_index(descriptor_q, num_neighbors=k, checks=64)
        
        if k == 1:
            if isinstance(indices, (int, np.integer)):
                indices = [indices]
            else:
                indices = indices.flatten().tolist()
        else:
            # Para k > 1, 'indices' debería ser una lista de listas
            indices = indices.flatten().tolist()
        
        for idx in indices:
            r_meta = descriptores_R[idx]
            resultados.append([
                q['archivo'],
                q['inicio'],
                r_meta['archivo'],
                r_meta['inicio']
            ])

    print(f"Escribiendo resultados en {archivo_ventanas_similares}...")
    escribir_lista_de_columnas_en_archivo(resultados, archivo_ventanas_similares)
    print("Búsqueda completada.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python tarea2-busqueda.py [carpeta_descriptores_radio_Q] [carpeta_descritores_canciones_R] [archivo_ventanas_similares]")
        sys.exit(1)
    
    carpeta_descriptores_Q = sys.argv[1]
    carpeta_descriptores_R = sys.argv[2]
    archivo_ventanas_similares = sys.argv[3]
    
    tarea2_busqueda(carpeta_descriptores_Q, carpeta_descriptores_R, archivo_ventanas_similares, k=10)
