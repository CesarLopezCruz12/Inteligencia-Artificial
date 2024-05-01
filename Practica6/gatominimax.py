# Función para imprimir el tablero
def imprimir_tablero(tablero):
    for fila in tablero:
        print("|".join(fila))
        print("-" * 5)

# Función para verificar si hay un ganador
def verificar_ganador(tablero, jugador):
    # Verificar filas y columnas
    for i in range(3):
        if all(tablero[i][j] == jugador for j in range(3)) or \
           all(tablero[j][i] == jugador for j in range(3)):
            return True

    # Verificar diagonales
    if all(tablero[i][i] == jugador for i in range(3)) or \
       all(tablero[i][2-i] == jugador for i in range(3)):
        return True

    return False

# Función para verificar si hay empate
def verificar_empate(tablero):
    return all(tablero[i][j] != " " for i in range(3) for j in range(3))

# Función para obtener los movimientos posibles
def obtener_movimientos(tablero):
    return [(i, j) for i in range(3) for j in range(3) if tablero[i][j] == " "]

# Función para evaluar el tablero
def evaluar_tablero(tablero):
    if verificar_ganador(tablero, "X"):
        return 1
    elif verificar_ganador(tablero, "O"):
        return -1
    else:
        return 0

# Función Minimax
def minimax(tablero, jugador):
    if verificar_ganador(tablero, "X"):
        return 1
    elif verificar_ganador(tablero, "O"):
        return -1
    elif verificar_empate(tablero):
        return 0

    movimientos = obtener_movimientos(tablero)
    if jugador == "X":
        mejor_valor = float("-inf")
        for movimiento in movimientos:
            i, j = movimiento
            tablero[i][j] = jugador
            valor = minimax(tablero, "O")
            tablero[i][j] = " "
            mejor_valor = max(mejor_valor, valor)
    else:
        mejor_valor = float("inf")
        for movimiento in movimientos:
            i, j = movimiento
            tablero[i][j] = jugador
            valor = minimax(tablero, "X")
            tablero[i][j] = " "
            mejor_valor = min(mejor_valor, valor)

    return mejor_valor

# Función para que la IA realice su movimiento
def movimiento_IA(tablero):
    mejor_movimiento = None
    mejor_valor = float("-inf")
    for movimiento in obtener_movimientos(tablero):
        i, j = movimiento
        tablero[i][j] = "X"
        valor = minimax(tablero, "O")
        tablero[i][j] = " "
        if valor > mejor_valor:
            mejor_valor = valor
            mejor_movimiento = movimiento
    i, j = mejor_movimiento
    tablero[i][j] = "X"

# Función principal para iniciar el juego
def jugar():
    tablero = [[" " for _ in range(3)] for _ in range(3)]
    jugador_actual = "O"

    while True:
        imprimir_tablero(tablero)
        if verificar_ganador(tablero, "X"):
            print("¡La IA gana!")
            break
        elif verificar_ganador(tablero, "O"):
            print("¡Has ganado!")
            break
        elif verificar_empate(tablero):
            print("¡Empate!")
            break

        if jugador_actual == "O":
            fila = int(input("Ingrese la fila (0-2): "))
            columna = int(input("Ingrese la columna (0-2): "))
            if tablero[fila][columna] != " ":
                print("Casilla ocupada, intenta nuevamente.")
                continue
            tablero[fila][columna] = "O"
        else:
            movimiento_IA(tablero)
        jugador_actual = "X" if jugador_actual == "O" else "O"

if __name__ == "__main__":
    jugar()
