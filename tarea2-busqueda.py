# tarea2-busqueda.py
# CC5213 - TAREA 2 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 20 septiembre de 2024
# Alumno: [Juan Vicente Onetto]

import sys
import os
import util as util
import numpy as np
import logging
import librosa
import scipy.spatial

# Configurar el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Ventana:
    def __init__(self, nombre_archivo, segundos_desde, segundos_hasta):
        self.nombre_archivo = nombre_archivo
        self.segundos_desde = segundos_desde
        self.segundos_hasta = segundos_hasta

    def __str__(self):
        return "{} [{:6.3f}-{:6.3f}]".format(self.nombre_archivo, self.segundos_desde, self.segundos_hasta)

def lista_ventanas(nombre_archivo, numero_descriptores, sample_rate, hop_length, n_fft):
    ventanas = []
    for i in range(numero_descriptores):
        # Tiempo de inicio y fin de la ventana
        segundos_desde = (i * hop_length) / sample_rate
        segundos_hasta = (i * hop_length + n_fft) / sample_rate
        ventana = Ventana(nombre_archivo, segundos_desde, segundos_hasta)
        ventanas.append(ventana)
    return ventanas

def tarea2_busqueda(carpeta_descriptores_radio_Q, carpeta_descritores_canciones_R, archivo_ventanas_similares):
    if not os.path.isdir(carpeta_descriptores_radio_Q):
        print("ERROR: no existe {}".format(carpeta_descriptores_radio_Q))
        sys.exit(1)
    elif not os.path.isdir(carpeta_descritores_canciones_R):
        print("ERROR: no existe {}".format(carpeta_descritores_canciones_R))
        sys.exit(1)
    elif os.path.exists(archivo_ventanas_similares):
        print("ERROR: ya existe {}".format(archivo_ventanas_similares))
        sys.exit(1)

    # Parámetros utilizados en tarea2-extractor.py
    sample_rate = 44100
    n_fft = 2048
    hop_length = 512
    dimension = 20  # Dimensión de los MFCC

    # 1. Leer descriptores de R (canciones)
    logging.info('Leyendo descriptores de canciones (R)')
    descriptores_R = []
    ventanas_R = []
    archivos_R = util.listar_archivos_con_extension(carpeta_descritores_canciones_R, '.mfcc')
    for archivo in archivos_R:
        descriptor = util.leer_objeto(carpeta_descritores_canciones_R, archivo)
        num_descriptores = descriptor.shape[0]
        ventanas = lista_ventanas(archivo, num_descriptores, sample_rate, hop_length, n_fft)
        descriptores_R.append(descriptor)
        ventanas_R.extend(ventanas)
    descriptores_R = np.vstack(descriptores_R)
    logging.info('Finalizado lectura de descriptores de R')

    # 2. Leer descriptores de Q (radio)
    logging.info('Leyendo descriptores de radio (Q)')
    descriptores_Q = []
    ventanas_Q = []
    archivos_Q = util.listar_archivos_con_extension(carpeta_descriptores_radio_Q, '.mfcc')
    for archivo in archivos_Q:
        descriptor = util.leer_objeto(carpeta_descriptores_radio_Q, archivo)
        num_descriptores = descriptor.shape[0]
        ventanas = lista_ventanas(archivo, num_descriptores, sample_rate, hop_length, n_fft)
        descriptores_Q.append(descriptor)
        ventanas_Q.extend(ventanas)
    descriptores_Q = np.vstack(descriptores_Q)
    logging.info('Finalizado lectura de descriptores de Q')

    # 3. Construir KDTree con descriptores de R
    logging.info('Construyendo KDTree para descriptores de R')
    tree_R = scipy.spatial.cKDTree(descriptores_R)

    # 4. Encontrar el vecino más cercano de cada descriptor de Q en R
    logging.info('Buscando vecinos más cercanos')
    distancias_min, indices_min = tree_R.query(descriptores_Q, k=1)

    # Asegurar que el directorio de salida existe
    output_dir = os.path.dirname(archivo_ventanas_similares)
    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)

    # 5. Escribir resultados en archivo_ventanas_similares
    logging.info('Escribiendo resultados en {}'.format(archivo_ventanas_similares))
    with open(archivo_ventanas_similares, 'w') as f_out:
        for i in range(len(ventanas_Q)):
            ventana_Q = ventanas_Q[i]
            ventana_R = ventanas_R[indices_min[i]]
            distancia = distancias_min[i]

            # Formato: archivo_Q \t tiempo_inicio_Q \t archivo_R \t tiempo_inicio_R \t distancia
            linea = "{}\t{}\t{}\t{}\t{}\n".format(
                ventana_Q.nombre_archivo,
                ventana_Q.segundos_desde,
                ventana_R.nombre_archivo,
                ventana_R.segundos_desde,
                distancia
            )
            f_out.write(linea)
    logging.info('Proceso completado exitosamente')

# Inicio de la tarea
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(
            "Uso: {} [carpeta_descriptores_radio_Q] [carpeta_descritores_canciones_R] [archivo_ventanas_similares]".format(
                sys.argv[0]))
        sys.exit(1)

    # Leer los parámetros de entrada
    carpeta_descriptores_radio_Q = sys.argv[1]
    carpeta_descritores_canciones_R = sys.argv[2]
    archivo_ventanas_similares = sys.argv[3]

    # Llamar a la tarea
    tarea2_busqueda(carpeta_descriptores_radio_Q, carpeta_descritores_canciones_R, archivo_ventanas_similares)
