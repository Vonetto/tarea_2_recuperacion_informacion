# tarea2-deteccion.py
import sys
import os
import util as util
import logging
from operator import itemgetter
from sklearn.cluster import DBSCAN
import numpy as np
import math

# Configurar el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_iou(det1, det2):
    if det1[0] != det2[0]:
        return 0.0
    start1, duration1 = det1[1], det1[2]
    start2, duration2 = det2[1], det2[2]
    end1 = start1 + duration1
    end2 = start2 + duration2
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    if intersection_start >= intersection_end:
        return 0.0
    intersection = intersection_end - intersection_start
    union = (end1 - start1) + (end2 - start2) - intersection
    return intersection / union

def non_maximum_suppression(detecciones, iou_threshold=0.5):
    detecciones = sorted(detecciones, key=itemgetter(4), reverse=True)
    final_detecciones = []
    while detecciones:
        current = detecciones.pop(0)
        final_detecciones.append(current)
        detecciones = [det for det in detecciones if calculate_iou(current, det) < iou_threshold]
    return final_detecciones

def cluster_detecciones(ventanas_similares, eps=2.5, min_samples=3):
    if not ventanas_similares:
        return []
    X = np.array([
        [
            ventana['ventana_Q']['tiempo_inicio'] - ventana['ventana_R']['tiempo_inicio'], 
            ventana['distancia']
        ] 
        for ventana in ventanas_similares
    ])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_
    detecciones = []
    for label in set(labels):
        if label == -1:
            continue  # Ignorar ruido
        cluster_indices = np.where(labels == label)[0]
        cluster = [ventanas_similares[i] for i in cluster_indices]
        archivo_Q = cluster[0]['ventana_Q']['archivo'].replace('.mfcc', '')
        archivo_R = cluster[0]['ventana_R']['archivo'].replace('.mfcc', '')
        tiempo_inicio_Q = min([ventana['ventana_Q']['tiempo_inicio'] for ventana in cluster])
        tiempo_fin_Q = max([ventana['ventana_Q']['tiempo_inicio'] for ventana in cluster])
        duracion = tiempo_fin_Q - tiempo_inicio_Q + 0.5
        confianza_promedio = np.mean([1 / (ventana['distancia'] + 1e-6) for ventana in cluster])
        detecciones.append([
            archivo_Q, 
            tiempo_inicio_Q, 
            duracion, 
            archivo_R, 
            confianza_promedio
        ])
    return detecciones

def tarea2_deteccion(archivo_ventanas_similares, archivo_detecciones, tolerancia_desfase=1.8, tolerancia_ventana=0.4, duracion_minima=1.3, iou_threshold=0.5, umbral_confianza=0.05):
    if not os.path.isfile(archivo_ventanas_similares):
        print("ERROR: no existe archivo {}".format(archivo_ventanas_similares))
        sys.exit(1)
    elif os.path.exists(archivo_detecciones):
        print("ERROR: ya existe archivo {}".format(archivo_detecciones))
        sys.exit(1)

    # 1. Leer el archivo archivo_ventanas_similares
    logging.info('Leyendo archivo de ventanas similares')
    ventanas_similares = []
    with open(archivo_ventanas_similares, 'r') as f_in:
        for linea in f_in:
            partes = linea.strip().split('\t')
            if len(partes) != 5:
                continue  # O manejar el error si el formato es incorrecto
            ventana_Q = {
                'archivo': partes[0],
                'tiempo_inicio': float(partes[1])
            }
            ventana_R = {
                'archivo': partes[2],
                'tiempo_inicio': float(partes[3])
            }
            distancia = float(partes[4])
            ventanas_similares.append({
                'ventana_Q': ventana_Q,
                'ventana_R': ventana_R,
                'distancia': distancia
            })
    logging.info(f'Se han leído {len(ventanas_similares)} ventanas similares')

    # 2. Crear un algoritmo para buscar secuencias similares entre audios
    detecciones = []
    if not ventanas_similares:
        logging.info('No se encontraron ventanas similares para procesar')
        return

    # Ordenar las ventanas por archivo Q y tiempo de inicio
    ventanas_similares.sort(key=lambda x: (x['ventana_Q']['archivo'], x['ventana_Q']['tiempo_inicio']))

    secuencia_actual = None

    for ventana in ventanas_similares:
        archivo_Q = ventana['ventana_Q']['archivo']
        tiempo_inicio_Q = ventana['ventana_Q']['tiempo_inicio']
        archivo_R = ventana['ventana_R']['archivo']
        tiempo_inicio_R = ventana['ventana_R']['tiempo_inicio']
        distancia = ventana['distancia']

        # Calcular el desfase actual entre Q y R
        desfase_actual = tiempo_inicio_Q - tiempo_inicio_R

        if secuencia_actual is None:
            # Iniciar una nueva secuencia
            secuencia_actual = {
                'archivo_Q': archivo_Q,
                'archivo_R': archivo_R,
                'desfase': desfase_actual,
                'tiempo_inicio_Q': tiempo_inicio_Q,
                'tiempo_fin_Q': tiempo_inicio_Q,
                'confianzas': [1 / (distancia + 1e-6)],
                'distancias': [distancia],
                'num_ventanas': 1
            }
        else:
            # Verificar si la ventana actual continúa la secuencia
            mismo_archivo_Q = archivo_Q == secuencia_actual['archivo_Q']
            mismo_archivo_R = archivo_R == secuencia_actual['archivo_R']
            desfase_similar = abs(desfase_actual - secuencia_actual['desfase']) <= tolerancia_desfase
            ventanas_consecutivas = tiempo_inicio_Q - secuencia_actual['tiempo_fin_Q'] <= tolerancia_ventana

            if mismo_archivo_Q and mismo_archivo_R and desfase_similar and ventanas_consecutivas:
                # Continuar la secuencia actual
                secuencia_actual['tiempo_fin_Q'] = tiempo_inicio_Q
                secuencia_actual['confianzas'].append(1 / (distancia + 1e-6))
                secuencia_actual['distancias'].append(distancia)
                secuencia_actual['num_ventanas'] += 1
            else:
                # Guardar la secuencia actual si tiene suficiente duración
                duracion = secuencia_actual['tiempo_fin_Q'] - secuencia_actual['tiempo_inicio_Q'] + 0.5  # Añadir un pequeño margen
                if duracion >= duracion_minima:
                    # Método Mejorado: Promedio Ponderado con Penalización Logarítmica
                    confianza_promedio = sum([
                        confianza * math.log(1 + 1 / (dist + 1e-6)) 
                        for confianza, dist in zip(secuencia_actual['confianzas'], secuencia_actual['distancias'])
                    ]) / sum([
                        math.log(1 + 1 / (dist + 1e-6)) 
                        for dist in secuencia_actual['distancias']
                    ])
                    confianza_promedio = min(confianza_promedio, 1.0)  # Limitar a 1.0

                    deteccion = [
                        secuencia_actual['archivo_Q'].replace('.mfcc', ''),
                        secuencia_actual['tiempo_inicio_Q'],
                        duracion,
                        secuencia_actual['archivo_R'].replace('.mfcc', ''),
                        confianza_promedio
                    ]
                    detecciones.append(deteccion)

                # Iniciar una nueva secuencia
                secuencia_actual = {
                    'archivo_Q': archivo_Q,
                    'archivo_R': archivo_R,
                    'desfase': desfase_actual,
                    'tiempo_inicio_Q': tiempo_inicio_Q,
                    'tiempo_fin_Q': tiempo_inicio_Q,
                    'confianzas': [1 / (distancia + 1e-6)],
                    'distancias': [distancia],
                    'num_ventanas': 1
                }

    # Procesar la última secuencia
    if secuencia_actual is not None:
        duracion = secuencia_actual['tiempo_fin_Q'] - secuencia_actual['tiempo_inicio_Q'] + 0.5
        if duracion >= duracion_minima:
            confianza_promedio = sum([
                confianza * math.log(1 + 1 / (dist + 1e-6)) 
                for confianza, dist in zip(secuencia_actual['confianzas'], secuencia_actual['distancias'])
            ]) / sum([
                math.log(1 + 1 / (dist + 1e-6)) 
                for dist in secuencia_actual['distancias']
            ])
            confianza_promedio = min(confianza_promedio, 1.0)

            deteccion = [
                secuencia_actual['archivo_Q'].replace('.mfcc', ''),
                secuencia_actual['tiempo_inicio_Q'],
                duracion,
                secuencia_actual['archivo_R'].replace('.mfcc', ''),
                confianza_promedio
            ]
            detecciones.append(deteccion)

    logging.info(f'Se han encontrado {len(detecciones)} detecciones antes de NMS')

    # 3. Aplicar Clustering (Opcional)
    detecciones_cluster = cluster_detecciones(ventanas_similares, eps=1.0, min_samples=5)
    logging.info(f'Se han encontrado {len(detecciones_cluster)} detecciones después de clustering')

    # 4. Aplicar Non-Maximum Suppression (NMS) para reducir duplicados y falsas positivas
    detecciones_nms = non_maximum_suppression(detecciones_cluster, iou_threshold=iou_threshold)
    logging.info(f'Se han encontrado {len(detecciones_nms)} detecciones después de NMS')

    # 5. Filtrar detecciones por umbral de confianza
    detecciones_finales = [det for det in detecciones_nms if det[4] >= umbral_confianza]
    logging.info(f'Se han encontrado {len(detecciones_finales)} detecciones después de filtrar por confianza')

    # 6. Escribir las detecciones encontradas en archivo_detecciones
    util.escribir_lista_de_columnas_en_archivo(detecciones_finales, archivo_detecciones)
    logging.info(f'Detecciones escritas en {archivo_detecciones}')

def evaluar_detecciones(detecciones, archivo_gt):
    """
    Implementa tu lógica de evaluación aquí.
    Debes comparar las detecciones con el Ground Truth y calcular precisión, recall y F1.
    """
    # Ejemplo simplificado (debes adaptarlo a tu caso específico)
    # Cargar el GT
    gt = []
    with open(archivo_gt, 'r') as f_gt:
        for linea in f_gt:
            partes = linea.strip().split('\t')
            if len(partes) != 5:
                continue
            etiqueta, archivo_q, tiempo_inicio, duracion, archivo_r = partes
            gt.append({
                'archivo_q': archivo_q,
                'tiempo_inicio': float(tiempo_inicio),
                'duracion': float(duracion),
                'archivo_r': archivo_r
            })

    # Comparar detecciones con GT
    correctas = 0
    detecciones_asignadas = set()
    for det in detecciones:
        archivo_q, tiempo_inicio_q, duracion, archivo_r, confianza = det
        for idx, g in enumerate(gt):
            if idx in detecciones_asignadas:
                continue  # Ya asignada a una detección
            if archivo_q != g['archivo_q'] or archivo_r != g['archivo_r']:
                continue
            iou = calculate_iou(det, [g['archivo_q'], g['tiempo_inicio'], g['duracion'], g['archivo_r'], 1.0])
            if iou >= 0.5:  # Umbral de IoU para considerar correcta la detección
                correctas += 1
                detecciones_asignadas.add(idx)
                break

    precision = correctas / len(detecciones) if detecciones else 0
    recall = correctas / len(gt) if gt else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

# Inicio de la tarea
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Uso: {} [archivo_ventanas_similares] [archivo_detecciones]".format(sys.argv[0]))
        sys.exit(1)

    # Leer los parámetros de entrada
    archivo_ventanas_similares = sys.argv[1]
    archivo_detecciones = sys.argv[2]

    # Parámetros ajustables
    tolerancia_desfase = 1.8
    tolerancia_ventana = 0.4
    duracion_minima = 1.3
    iou_threshold = 0.6  # Ajustado
    umbral_confianza = 0.02

    # Llamar a la tarea
    tarea2_deteccion(
        archivo_ventanas_similares, 
        archivo_detecciones, 
        tolerancia_desfase, 
        tolerancia_ventana, 
        duracion_minima, 
        iou_threshold, 
        umbral_confianza
    )
