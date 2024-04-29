import numpy as np 


# Nombre del archivo .data
nombre_archivo = 'bezdekIris.data'

# Leer la matriz desde el archivo .data
with open(nombre_archivo, 'r') as archivo:
    matriz = archivo.readlines()

# Inicializar listas para almacenar valores numéricos
columnas = [[] for _ in range(len(matriz[0].strip().split(','))-1)]

# Llenar las listas con valores numéricos
for fila in matriz:
    valores = fila.strip().split(',')[:-1]  # Ignorar la última columna
    for i, valor in enumerate(valores):
        columnas[i].append(float(valor))

# Calcular promedio, varianza y desviación estándar por columna
for i, columna in enumerate(columnas):
    promedio = np.mean(columna)
    varianza = np.var(columna)
    desviacion_estandar = np.std(columna)
    
    print(f"Columna {i+1}:")
    print(f"   Promedio: {promedio}")
    print(f"   Varianza: {varianza}")
    print(f"   Desviación estándar: {desviacion_estandar}")


# Inicializar un diccionario para almacenar los datos agrupados por categoría
datos_por_categoria = {}

# Agrupar los datos por categoría
for linea in matriz:
    valores = linea.strip().split(',')
    categoria = valores[-1]  # Última columna es la categoría
    datos = list(map(float, valores[:-1]))  # Convertir valores a flotantes, ignorando la categoría
    if categoria not in datos_por_categoria:
        datos_por_categoria[categoria] = []
    datos_por_categoria[categoria].append(datos)

# Calcular promedio, varianza y desviación estándar por columna para cada categoría
for categoria, datos_categoria in datos_por_categoria.items():
    print(f"Categoría: {categoria}")
    datos_categoria = np.array(datos_categoria)  # Convertir a una matriz numpy para cálculos más fáciles
    promedio = np.mean(datos_categoria, axis=0)
    varianza = np.var(datos_categoria, axis=0)
    desviacion_estandar = np.std(datos_categoria, axis=0)
    
    for i in range(len(promedio)):
        print(f"   Columna {i+1}:")
        print(f"      Promedio: {promedio[i]}")
        print(f"      Varianza: {varianza[i]}")
        print(f"      Desviación estándar: {desviacion_estandar[i]}")