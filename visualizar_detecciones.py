import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def leer_gt(archivo_gt):
    """
    Lee el archivo de Ground Truth y lo devuelve como un DataFrame.
    """
    columnas = ['Etiqueta', 'archivo_q', 'tiempo_inicio', 'duracion', 'archivo_r']
    df_gt = pd.read_csv(archivo_gt, sep='\t', header=None, names=columnas)
    return df_gt

def leer_detecciones(archivo_detecciones):
    """
    Lee el archivo de detecciones y lo devuelve como un DataFrame.
    """
    columnas = ['archivo_q', 'tiempo_inicio', 'duracion', 'archivo_r', 'confianza_promedio']
    
    # Intentar leer como tabulado
    try:
        df_det = pd.read_csv(archivo_detecciones, sep='\t', header=None, names=columnas)
        if 'tiempo_inicio' not in df_det.columns:
            raise ValueError("Columna 'tiempo_inicio' no encontrada al usar tabulaciones.")
    except Exception as e:
        print(f"Error al leer detecciones con sep='\t': {e}")
        print("Intentando leer detecciones con sep='\s+' (espacios).")
        # Intentar leer como espacio
        try:
            df_det = pd.read_csv(archivo_detecciones, sep='\s+', header=None, names=columnas)
            if 'tiempo_inicio' not in df_det.columns:
                raise ValueError("Columna 'tiempo_inicio' no encontrada al usar espacios.")
        except Exception as e2:
            print(f"Error al leer detecciones con sep='\s+': {e2}")
            print("Revisa el formato de tu archivo de detecciones.")
            exit(1)
    
    return df_det

def agrupar_por_archivo(df, archivo_col):
    """
    Agrupa las detecciones o GT por el archivo_q.
    """
    grupos = df.groupby(archivo_col)
    return grupos

def plot_timeline(archivo, detecciones, gt, ax):
    """
    Grafica las detecciones y GT en una línea de tiempo para un archivo específico.
    """
    # Plot GT
    for _, row in gt.iterrows():
        ax.broken_barh([(row['tiempo_inicio'], row['duracion'])], (10, 9),
                      facecolors='green', edgecolors='black')
    
    # Plot Detecciones
    for _, row in detecciones.iterrows():
        ax.broken_barh([(row['tiempo_inicio'], row['duracion'])], (20, 9),
                      facecolors='red', edgecolors='black', alpha=0.5)
    
    # Ajustes del gráfico
    ax.set_ylim(5, 35)
    max_det = detecciones['tiempo_inicio'].max() + detecciones['duracion'].max() if not detecciones.empty else 0
    max_gt = gt['tiempo_inicio'].max() + gt['duracion'].max() if not gt.empty else 0
    ax.set_xlim(0, max(max_det, max_gt) + 10)
    ax.set_xlabel('Tiempo (s)')
    ax.set_yticks([15, 25])
    ax.set_yticklabels(['Ground Truth', 'Detecciones'])
    ax.set_title(f'Detecciones vs Ground Truth para {archivo}')

def main_especifico(archivo_gt, archivo_detecciones, archivo_especifico, salida_grafico='comparacion_detecciones_gt_especifico.png'):
    # Leer los datos
    df_gt = leer_gt(archivo_gt)
    df_det = leer_detecciones(archivo_detecciones)
    
    # Filtrar por archivo_especifico
    df_gt_filtrado = df_gt[df_gt['archivo_q'] == archivo_especifico]
    df_det_filtrado = df_det[df_det['archivo_q'] == archivo_especifico]
    
    # Configurar el tamaño del gráfico
    plt.figure(figsize=(15, 3))
    ax = plt.subplot(1, 1, 1)
    
    plot_timeline(archivo_especifico, df_det_filtrado, df_gt_filtrado, ax)
    
    # Crear leyenda
    legend_patches = [
        mpatches.Patch(color='green', label='Ground Truth'),
        mpatches.Patch(color='red', alpha=0.5, label='Detecciones')
    ]
    plt.legend(handles=legend_patches, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(salida_grafico)
    plt.show()

if __name__ == "__main__":
    # Rutas de los archivos (ajusta según tu estructura de carpetas)
    archivo_gt = 'gt.txt'  # Ruta al archivo de Ground Truth
    archivo_detecciones = 'evaluacion_tarea2/dataset_a/resultados.dataset_a.txt'  # Ruta al archivo de detecciones
    archivo_especifico = 'radio-disney-ar-1.m4a'  # Especifica el archivo de audio que deseas visualizar
    salida_grafico = 'comparacion_detecciones_gt_especifico.png'  # Nombre del archivo de salida del gráfico
    
    # Verificar que los archivos existan
    if not os.path.isfile(archivo_gt):
        print(f"ERROR: No se encuentra el archivo de Ground Truth en {archivo_gt}")
        exit(1)
    if not os.path.isfile(archivo_detecciones):
        print(f"ERROR: No se encuentra el archivo de detecciones en {archivo_detecciones}")
        exit(1)
    
    # Ejecutar la función para un archivo específico
    main_especifico(archivo_gt, archivo_detecciones, archivo_especifico, salida_grafico)

if __name__ == "__main__":
    # Rutas de los archivos (ajusta según tu estructura de carpetas)
    archivo_gt = './datasets/dataset_a/gt.txt' # Ruta al archivo de Ground Truth
    
    archivo_detecciones = './resultados/detecciones.txt'  # Ruta al archivo de detecciones
    salida_grafico = 'comparacion_detecciones_gt.png'  # Nombre del archivo de salida del gráfico
    
    # Verificar que los archivos existan
    if not os.path.isfile(archivo_gt):
        print(f"ERROR: No se encuentra el archivo de Ground Truth en {archivo_gt}")
        exit(1)
    if not os.path.isfile(archivo_detecciones):
        print(f"ERROR: No se encuentra el archivo de detecciones en {archivo_detecciones}")
        exit(1)
    
    # Ejecutar la función principal
    main_especifico(archivo_gt, archivo_detecciones, archivo_especifico, salida_grafico)
