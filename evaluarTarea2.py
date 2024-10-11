# CC5213 - TAREA 2 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 20 septiembre de 2024
# Profesor: Juan Manuel Barrios

import sys
import os
import numpy
import shutil
import time
import subprocess


class Deteccion:
    def __init__(self, id_deteccion, archivo_fuente, tipo, radio, desde, largo, cancion, confianza):
        self.id_deteccion = id_deteccion
        self.archivo_fuente = archivo_fuente
        self.tipo = tipo
        self.radio = radio
        self.desde = desde
        self.largo = largo
        self.cancion = cancion
        self.confianza = confianza

    def interseccion(self, otra):
        if self.radio != otra.radio or self.cancion != otra.cancion:
            return 0
        ini1 = self.desde
        end1 = self.desde + self.largo
        ini2 = otra.desde
        end2 = otra.desde + otra.largo
        inter = min(end1, end2) - max(ini1, ini2)
        union = max(end1, end2) - min(ini1, ini2)
        if inter <= 0 or union <= 0:
            return 0
        return inter / union


def get_filename(filepath):
    name = filepath.lower().strip()
    if name.rfind('/') >= 0:
        name = name[name.rfind('/') + 1:]
    if name.rfind('\\') >= 0:
        name = name[name.rfind('\\') + 1:]
    return name


def parsear_deteccion(id_deteccion, archivo_fuente, linea, es_gt):
    linea = linea.rstrip("\r\n")
    # se ignoran lineas vacias o comentarios
    if linea == "" or linea.startswith("#"):
        return None
    partes = linea.split("\t")
    if len(partes) != 5:
        raise Exception(
            archivo_fuente + " incorrecto numero de columnas (se esperan 5 columnas separadas por un tabulador)")
    tipo = radio = cancion = ""
    desde = largo = confianza = 0
    if es_gt:
        tipo = partes[0]
        radio = get_filename(partes[1])
        desde = round(float(partes[2]), 3)
        largo = round(float(partes[3]), 3)
        cancion = get_filename(partes[4])
    else:
        radio = get_filename(partes[0])
        desde = round(float(partes[1]), 3)
        largo = round(float(partes[2]), 3)
        cancion = get_filename(partes[3])
        confianza = float(partes[4])
        if confianza <= 0:
            raise Exception("valor incorrecto confianza={} en {}".format(confianza, archivo_fuente))
    if radio == "":
        raise Exception("nombre radio invalido en " + archivo_fuente)
    if cancion == "":
        raise Exception("nombre cancion invalido en " + archivo_fuente)
    if desde < 0:
        raise Exception("valor incorrecto desde={} en {}".format(desde, archivo_fuente))
    if largo <= 0:
        raise Exception("valor incorrecto largo={} en {}".format(largo, archivo_fuente))
    det = Deteccion(id_deteccion, archivo_fuente, tipo, radio, desde, largo, cancion, confianza)
    return det


def leer_archivo_detecciones(lista, filename, es_gt):
    if not os.path.isfile(filename):
        if filename == "":
            return
        raise Exception("no existe el archivo {}".format(filename))
    cont_lineas = 0
    cont_detecciones = 0
    with open(filename) as f:
        for linea in f:
            cont_lineas += 1
            try:
                # el id es su posición en la lista
                det = parsear_deteccion(len(lista), filename, linea, es_gt)
                if det is not None:
                    lista.append(det)
                    cont_detecciones += 1
            except Exception as ex:
                print("Error {} (linea {}): {}".format(filename, cont_lineas, ex))
    print("{} detecciones en archivo {}".format(cont_detecciones, filename))


class ResultadoDeteccion:
    def __init__(self, deteccion):
        self.deteccion = deteccion
        self.es_incorrecta = False
        self.es_duplicada_misma_fuente = False
        self.es_duplicada_otra_fuente = False
        self.es_correcta = False
        self.gt = None
        self.iou = 0


class Metricas:
    def __init__(self, threshold):
        self.threshold = threshold
        self.total_gt = 0
        self.total_detecciones = 0
        self.correctas = 0
        self.incorrectas = 0
        self.recall = 0
        self.precision = 0
        self.f1 = 0
        self.iou = 0
        self.f1_iou = 0
        self.correctas_por_tipo = dict()
        self.recall_por_tipo = dict()


class Evaluacion:
    def __init__(self):
        self.detecciones_gt = list()
        self.total_gt_por_tipo = dict()
        self.detecciones = list()
        self.resultado_por_deteccion = list()
        self.resultado_global = None

    def leer_archivo_gt(self, file_gt):
        # cargar el ground-truth
        leer_archivo_detecciones(self.detecciones_gt, file_gt, True)
        for gt in self.detecciones_gt:
            self.total_gt_por_tipo[gt.tipo] = self.total_gt_por_tipo.get(gt.tipo, 0) + 1

    def leer_archivo_detecciones(self, file_detecciones):
        # cargar las detecciones
        leer_archivo_detecciones(self.detecciones, file_detecciones, False)

    def evaluar_cada_deteccion(self):
        # ordenar detecciones por confianza de mayor a menor
        self.detecciones.sort(key=lambda x: x.confianza, reverse=True)
        # para descartar las detecciones duplicadas
        ids_encontradas = {}
        # revisar cada deteccion
        for det in self.detecciones:
            # evaluar cada deteccion si es correcta a no
            gt_encontrada, iou = self.buscar_deteccion_en_gt(det)
            # retorna resultado
            res = ResultadoDeteccion(det)
            if gt_encontrada is None:
                res.es_incorrecta = True
            elif gt_encontrada.id_deteccion in ids_encontradas:
                fuentes = ids_encontradas[gt_encontrada.id_deteccion]
                # revisar si ya se detectó en el mismo archivo
                if det.archivo_fuente in fuentes:
                    res.es_duplicada_misma_fuente = True
                else:
                    res.es_duplicada_otra_fuente = True
                    fuentes.add(det.archivo_fuente)
            else:
                res.es_correcta = True
                res.gt = gt_encontrada
                res.iou = iou
                # agrego el origen que lo detectó
                fuentes = set()
                fuentes.add(det.archivo_fuente)
                ids_encontradas[gt_encontrada.id_deteccion] = fuentes
            self.resultado_por_deteccion.append(res)
        # ordenar los resultados como el archivo de entrada
        self.resultado_por_deteccion.sort(key=lambda x: x.deteccion.id_deteccion)

    def buscar_deteccion_en_gt(self, deteccion):
        gt_encontrada = None
        iou = 0
        # busca en gt la deteccion que tiene mayor interseccion
        for det_gt in self.detecciones_gt:
            interseccion = deteccion.interseccion(det_gt)
            if interseccion > iou:
                gt_encontrada = det_gt
                iou = interseccion
        return gt_encontrada, iou

    def calcular_metricas(self):
        # todos los umbrales posibles
        set_confianzas = set()
        for res in self.resultado_por_deteccion:
            if res.es_correcta:
                set_confianzas.add(res.deteccion.confianza)
        set_confianzas.add(0)
        # calcular las metricas para cada confianza y seleccionar el mejor
        for confianza in sorted(list(set_confianzas), reverse=True):
            met = self.evaluar_con_threshold(confianza)
            if self.resultado_global is None or met.f1_iou > self.resultado_global.f1_iou:
                self.resultado_global = met

    def evaluar_con_threshold(self, threshold):
        met = Metricas(threshold)
        met.total_gt = len(self.detecciones_gt)
        suma_iou = 0
        correctas_por_tipo = dict()
        for res in self.resultado_por_deteccion:
            # ignorar detecciones con confianza bajo el umbral
            if res.deteccion.confianza < threshold or res.es_duplicada_otra_fuente:
                continue
            met.total_detecciones += 1
            if res.es_correcta:
                met.correctas += 1
                suma_iou += res.iou
                correctas_por_tipo[res.gt.tipo] = correctas_por_tipo.get(res.gt.tipo, 0) + 1
            if res.es_incorrecta or res.es_duplicada_misma_fuente:
                met.incorrectas += 1
        if met.correctas > 0:
            # recall mide lo detectado con respecto al total de detecciones
            met.recall = met.correctas / met.total_gt
            # precision mide la relacion entre detecciones correctas e incorrectas
            met.precision = met.correctas / met.total_detecciones
            # F1 combina precision con recall usando la media armónica
            met.f1 = (2 * met.precision * met.recall) / (met.precision + met.recall)
            # IoU (intersection over union) mide que tan exacto es el intervalo detectado
            met.iou = suma_iou / met.correctas
            # para evaluar se usa una combinacion 80% de F1 con 20% de IoU
            met.f1_iou = met.f1 * 0.8 + met.iou * 0.2
        for tipo in self.total_gt_por_tipo:
            total = self.total_gt_por_tipo[tipo]
            correctas = correctas_por_tipo.get(tipo, 0)
            met.correctas_por_tipo[tipo] = correctas
            met.recall_por_tipo[tipo] = correctas / total
        return met

    def imprimir_resultado_por_deteccion(self):
        if len(self.resultado_por_deteccion) == 0:
            return
        print("Resultado detallado de cada una de las {} detecciones:".format(len(self.resultado_por_deteccion)))
        for res in self.resultado_por_deteccion:
            s1 = ""
            s2 = ""
            if res.es_correcta:
                s1 = "   OK)"
                s2 = " //IoU={:.1%} gt=({} {})".format(res.iou, res.gt.desde, res.gt.largo)
            elif res.es_duplicada_misma_fuente:
                s1 = "dup--)"
            elif res.es_duplicada_otra_fuente:
                s1 = "dupOK)"
            elif res.es_incorrecta:
                s1 = "   --)"
            d = res.deteccion
            print(" {} {} {} {} {} {} {}".format(s1, d.radio, d.desde, d.largo, d.cancion, d.confianza, s2))

    def imprimir_resultado_global(self):
        if self.resultado_global is None:
            return
        m = self.resultado_global
        print()
        print("Resultado global:")
        print(" {} detecciones en GT, {} detecciones a evaluar".format(m.total_gt, len(self.resultado_por_deteccion)))
        print(" Al usar umbral={} se seleccionan {} detecciones:".format(m.threshold, m.total_detecciones))
        print("    {} detecciones correctas, {} detecciones incorrectas".format(m.correctas, m.incorrectas))
        print("    Precision={:.3f} ({}/{})  Recall={:.3f} ({}/{})".format(m.precision, m.correctas,
                                                                           m.total_detecciones, m.recall, m.correctas,
                                                                           m.total_gt))
        print("    F1={:.3f}  IoU={:.1%}  ->  F1-IOU={:.3f}".format(m.f1, m.iou, m.f1_iou))
        print()
        print("Resultado por transformacion:")
        for tipo in m.recall_por_tipo:
            print("    {:9s}={:4} correctas ({:.0f}%)".format(tipo, m.correctas_por_tipo[tipo],
                                                              100 * m.recall_por_tipo[tipo]))


def evaluar_resultado_en_dataset(filename_gt, filename_resultados):
    print()
    print("Evaluando {} con {}".format(filename_resultados, filename_gt))
    ev = Evaluacion()
    ev.leer_archivo_gt(filename_gt)
    ev.leer_archivo_detecciones(filename_resultados)
    ev.evaluar_cada_deteccion()
    ev.calcular_metricas()
    ev.imprimir_resultado_por_deteccion()
    ev.imprimir_resultado_global()
    return ev.resultado_global.f1_iou


def validar_tiempo_maximo(t0):
    segundos = time.time() - t0
    # el enunciado dice que no puede demorar mas de 15 minutos
    if segundos > 900:
        print("La tarea no puede demorar más de 15 minutos!!")
        sys.exit(1)


def ejecutar(comando):
    t0 = time.time()
    print()
    print("Ejecutando:")
    print("[{}] ".format(time.strftime("%d-%m-%Y %H:%M:%S")) + " ".join(comando))
    code = subprocess.call(comando)
    print()
    if code != 0:
        print("EL PROGRAMA RETORNA ERROR!")
        sys.exit(1)
    validar_tiempo_maximo(t0)


def ejecutar_tarea(nombre, carpeta_radio, carpeta_canciones, dir_evaluacion):
    datos_temporales = dir_evaluacion + "/" + nombre
    dir_descriptores_canciones = datos_temporales + "/descriptores_canciones/"
    dir_descriptores_radio = datos_temporales + "/descriptores_radio/"
    file_similares = datos_temporales + "/similares.{}.bin".format(nombre)
    file_detecciones = datos_temporales + "/resultados.{}.txt".format(nombre)
    # comando para calcular descriptores Q
    comando = [sys.executable, "tarea2-extractor.py", carpeta_radio, dir_descriptores_radio]
    ejecutar(comando)
    # comando para calcular descriptores R
    comando = [sys.executable, "tarea2-extractor.py", carpeta_canciones, dir_descriptores_canciones]
    ejecutar(comando)
    # comando para buscar
    comando = [sys.executable, "tarea2-busqueda.py", dir_descriptores_radio, dir_descriptores_canciones,
               file_similares]
    ejecutar(comando)
    # comando para detectar
    comando = [sys.executable, "tarea2-deteccion.py", file_similares, file_detecciones]
    ejecutar(comando)
    return file_detecciones


def evaluar_en_dataset(nombre, dir_evaluacion):
    dataset_basedir = "datasets/" + nombre
    if not os.path.isdir(dataset_basedir):
        print("no existe {}".format(dataset_basedir))
        sys.exit(1)
    carpeta_radio = dataset_basedir + "/radio/"
    carpeta_canciones = dataset_basedir + "/canciones/"
    archivo_gt = dataset_basedir + "/gt.txt"
    if not os.path.isdir(carpeta_radio):
        print("error leyendo {}. No existe {}".format(nombre, carpeta_radio))
        sys.exit(1)
    if not os.path.isdir(carpeta_canciones):
        print("error leyendo {}. No existe {}".format(nombre, carpeta_canciones))
        sys.exit(1)
    if not os.path.isfile(archivo_gt):
        print("error leyendo {}. No existe {}".format(nombre, archivo_gt))
        sys.exit(1)
    t0 = time.time()
    archivo_detecciones = ejecutar_tarea(nombre, carpeta_radio, carpeta_canciones, dir_evaluacion)
    validar_tiempo_maximo(t0)
    metricas = evaluar_resultado_en_dataset(archivo_gt, archivo_detecciones)
    return metricas


def calcular_nota(f1_promedio):
    f1_para_4 = 0.25
    f1_para_7 = 0.90
    f1_max_bonus = 0.98
    if f1_promedio <= f1_para_4:
        nota = 1 + round(3 * f1_promedio / f1_para_4, 1)
        bonus = 0
    elif f1_promedio <= f1_para_7:
        nota = 4 + round(3 * (f1_promedio - f1_para_4) / (f1_para_7 - f1_para_4), 1)
        bonus = 0
    elif f1_promedio <= f1_max_bonus:
        nota = 7
        bonus = round((f1_promedio - f1_para_7) / (f1_max_bonus - f1_para_7), 1)
    else:
        nota = 7
        bonus = 1
    return nota, bonus


def evaluar_tarea2(letras_datasets):
    print("CC5213 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA")
    print("Evaluación Tarea 2 - 2024")
    # datos para la evaluacion
    dir_evaluacion = "evaluacion_tarea2"
    if os.path.exists(dir_evaluacion):
        print("borrando datos previos en {}...".format(dir_evaluacion))
        shutil.rmtree(dir_evaluacion)
    # evaluar sobre los datasets
    resultados = {}
    for letra in letras_datasets:
        dataset_nombre = "dataset_" + letra
        print()
        print("------- EVALUACION EN: {} -------".format(dataset_nombre))
        t0 = time.time()
        resultado_f1 = evaluar_en_dataset(dataset_nombre, dir_evaluacion)
        segundos = time.time() - t0
        print("  tiempo: {:.1f} segundos".format(segundos))
        resultados[dataset_nombre] = (resultado_f1, segundos)
    print()
    print("--------------------------------------------")
    print("Resumen:")
    f1s = []
    for nombre in resultados:
        (f1, segundos) = resultados[nombre]
        f1s.append(f1)
        print("    F1-IOU en {}: {:.3f}   (tiempo={:.1f} segundos)".format(nombre, f1, segundos))
    f1_promedio = numpy.average(f1s)
    print("    ==> Promedio F1-IOU: {:.3f}".format(f1_promedio))
    nota, bonus = calcular_nota(f1_promedio)
    print()
    print("    ==> Nota tarea 2 = {:.1f}".format(nota))
    if bonus > 0:
        print("    ==> Bonus = {:.1f}".format(bonus))


# parametros de entrada
datasets = ["a", "b", "c", "d"]
if len(sys.argv) > 1:
    datasets = sys.argv[1:]

evaluar_tarea2(datasets)
