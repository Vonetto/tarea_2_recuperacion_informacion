# tarea2-extractor.py
# CC5213 - TAREA 2 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 20 septiembre de 2024
# Alumno: [Juan Vicente Onetto]

import sys
import os
import util as util
import librosa
import numpy as np
import logging

# Configurar el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calcular_mfcc(archivo_wav, sample_rate, n_fft, hop_length, n_mfcc):
    # Cargar el archivo de audio
    samples, sr = librosa.load(archivo_wav, sr=sample_rate, mono=True)
    logging.info(f'Audio cargado: {archivo_wav}, muestras: {len(samples)}, sample_rate: {sr}')

    # Calcular los MFCC
    mfcc = librosa.feature.mfcc(
        y=samples,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # Transponer para tener los descriptores como filas
    return mfcc.T

def tarea2_extractor(carpeta_audios_entrada, carpeta_descriptores_salida):
    if not os.path.isdir(carpeta_audios_entrada):
        print("ERROR: no existe {}".format(carpeta_audios_entrada))
        sys.exit(1)
    elif os.path.exists(carpeta_descriptores_salida):
        print("ERROR: ya existe {}".format(carpeta_descriptores_salida))
        sys.exit(1)

    # Parámetros para el cálculo de MFCC
    sample_rate = 44100
    n_fft = 2048
    hop_length = 512
    n_mfcc = 20  # Dimensión de los MFCC

    # Crear la carpeta de salida si no existe
    os.makedirs(carpeta_descriptores_salida, exist_ok=True)

    # 1. Leer los archivos con extensión .m4a en carpeta_audios_entrada
    archivos_m4a = util.listar_archivos_con_extension(carpeta_audios_entrada, ".m4a")
    logging.info(f'Archivos a procesar: {len(archivos_m4a)}')

    # 2. Convertir cada archivo de audio a WAV y calcular descriptores
    for archivo_m4a in archivos_m4a:
        ruta_entrada = os.path.join(carpeta_audios_entrada, archivo_m4a)
        # Convertir a WAV usando FFmpeg
        archivo_wav = util.convertir_a_wav(ruta_entrada, sample_rate, carpeta_descriptores_salida)
        # Calcular los MFCC
        descriptores = calcular_mfcc(archivo_wav, sample_rate, n_fft, hop_length, n_mfcc)
        # Guardar los descriptores
        nombre_descriptor = os.path.basename(archivo_m4a) + ".mfcc"
        util.guardar_objeto(descriptores, carpeta_descriptores_salida, nombre_descriptor)
        logging.info(f'Descriptores guardados para: {archivo_wav}')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Uso: {} [carpeta_audios_entrada] [carpeta_descriptores_salida]".format(sys.argv[0]))
        sys.exit(1)

    carpeta_audios_entrada = sys.argv[1]
    carpeta_descriptores_salida = sys.argv[2]

    tarea2_extractor(carpeta_audios_entrada, carpeta_descriptores_salida)
