import math
import random

# Función objetivo 1
def f1(x):
    return x**4 + 3*x**3 + 2*x**2 - 1

# Función objetivo 2
def f2(x):
    return x**2 - 3*x - 8

# Generación de vecino
def generar_vecino(x):
    return x + random.uniform(-0.1, 0.1)  # Pequeña perturbación aleatoria

# Criterio de aceptación
def aceptar(solucion_actual, solucion_vecina, temperatura):
    delta_costo = f(solucion_vecina) - f(solucion_actual)  #Escoger que función objetivo se requiere
    if delta_costo < 0:
        return True
    else:
        probabilidad_aceptacion = math.exp(-delta_costo / temperatura)
        return random.random() < probabilidad_aceptacion

# Enfriamiento
def enfriar(temperatura, tasa_enfriamiento):
    return temperatura * tasa_enfriamiento

# Algoritmo de templado simulado
def templado_simulado(f, temperatura_inicial, tasa_enfriamiento, iteraciones):
    solucion_actual = random.uniform(-10, 10)  # Solución inicial aleatoria
    temperatura = temperatura_inicial
    
    for _ in range(iteraciones):
        solucion_vecina = generar_vecino(solucion_actual)
        
        if aceptar(solucion_actual, solucion_vecina, temperatura):
            solucion_actual = solucion_vecina
        
        temperatura = enfriar(temperatura, tasa_enfriamiento)
    
    return solucion_actual

# Parámetros del algoritmo
temperatura_inicial = 100.0
tasa_enfriamiento = 0.90
iteraciones = 10000

# Ejecución del algoritmo para la función f1
solucion_optima_f1 = templado_simulado(f1, temperatura_inicial, tasa_enfriamiento, iteraciones)
print("Solución óptima para f(x) = x^4 + 3*x^3 + 2*x^2 - 1:", solucion_optima_f1)

# Ejecución del algoritmo para la función f2
solucion_optima_f2 = templado_simulado(f2, temperatura_inicial, tasa_enfriamiento, iteraciones)
print("Solución óptima para f(x) = x^2 - 3*x - 8:", solucion_optima_f2)
