# CC5213 - TAREA 2 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
## Profesor: Juan Manuel Barrios
## 20 septiembre de 2024

# Tarea 2

Para la tarea 2 debe crear tres programas:

  * `tarea2-extractor.py`
     Recibe dos parámetros por la línea de comandos:
	   1. Una carpeta con audios. Procesar todos los archivos .m4a en esa carpeta.
       2. Una carpeta donde guardar descriptores calculados.
   
  * `tarea2-busqueda.py`
     Recibe tres parámetros por la línea de comandos:
	   1. La carpeta con descriptores de Q (radio).
	   2. La carpeta con descriptores de R (canciones).
	   3. Un nombre de archivo donde guardar el resultado de la comparación de Q y R.

  * `tarea2-deteccion.py`
     Recibe tres parámetros por la línea de comandos:
	   1. El nombre del archivo con el resultado de la comparación de Q y R.
	   2. El archivo a crear con el resultado de la detección de canciones.

El archivo de salida debe tener un formato de 5 columnas separadas por tabulador. En cada
fila debe tener un archivo de Q (radio), un tiempo de inicio en segundos, un largo en
segundos, un archivo de R (canciones) y el valor de confianza de que la canción sea
audible en la radio en la ventana de tiempo señalada.

Por ejemplo, un posible archivo de resultados sería este:

radio-disney-ar-3.m4a	129.6	21.5	The Doors - Break on through (23).m4a	21
radio-disney-ar-3.m4a	813.8	12.6	Daft Punk - Around the world (13).m4a	32
radio-disney-ar-3.m4a	1559.2	40.4	Michael Jackson - Billie Jean (41).m4a	167
radio-disney-br-3.m4a	817.7	28.2	Celia Cruz - La Vida es un Carnaval (29).m4a	57
radio-disney-br-3.m4a	1363.6	32.5	Paul Anka - Diana (34).m4a	37
[......]


Para probar su tarea debe usar el programa de evaluación:

  `python evaluarTarea2.py`

Este programa llamará su tarea con todos los datasets ["a", "b", "c", "d"] y
mostrará el resultado obtenido y la nota.

Opcionalmente, para probar su tarea con un solo dataset puede usar:

  `python evaluarTarea1.py a`

Su tarea no puede demorar más de 15 minutos en evaluar cada dataset.
