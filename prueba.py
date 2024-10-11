import util as util

# Ruta al archivo generado por la tarea de búsqueda
archivo_ventanas_similares = './resultados/ventanas_similares.pkl'

# Cargar el archivo
ventanas_similares = util.leer_objeto('', archivo_ventanas_similares)

# Mostrar los primeros elementos para ver la estructura
for ventana in ventanas_similares[:5]:
    print(ventana)
