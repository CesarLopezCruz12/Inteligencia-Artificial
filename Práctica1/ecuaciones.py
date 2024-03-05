import numpy as np

def generar_ecuaciones(grado):
    coeficientes = np.random.randint(-100, 101, size=(grado+1, grado+1))
    return coeficientes[:, :-1], coeficientes[:, -1]

def resolver_sistema(ecuaciones, resultados):
    solucion = np.linalg.solve(ecuaciones, resultados)
    return solucion

def main():
    grado = int(input("Ingrese el grado del sistema de ecuaciones: "))
    ecuaciones, resultados = generar_ecuaciones(grado)
    print("Sistema de ecuaciones:")
    for i in range(grado):
        ecuacion = " + ".join([f"{ecuaciones[i][j]}x_{j}" for j in range(grado)])
        print(f"{ecuacion} = {resultados[i]}")

    intentos = 0
    while True:
        intentos += 1
        print(f"\nIntento #{intentos}:")
        solucion = resolver_sistema(ecuaciones, resultados)
        print("Solución encontrada:")
        for i in range(grado):
            print(f"x_{i} = {solucion[i]}")

        if np.allclose(np.dot(ecuaciones, solucion), resultados):
            print(f"\n¡La solución es correcta en {intentos} intento(s)!")
            break
        else:
            print("La solución encontrada no es válida.")

if __name__ == "__main__":
    main()
