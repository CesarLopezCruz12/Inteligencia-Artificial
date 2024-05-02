import copy

# Función para imprimir el tablero
def imprimir_tablero(tablero):
    for fila in tablero:
        print("|".join(fila))
    print()

# Función para verificar si alguien ha ganado
def verificar_ganador(tablero, jugador):
    # Verificar filas
    for fila in tablero:
        if all(casilla == jugador for casilla in fila):
            return True

    # Verificar columnas
    for col in range(3):
        if all(tablero[fila][col] == jugador for fila in range(3)):
            return True

    # Verificar diagonales
    if all(tablero[i][i] == jugador for i in range(3)) or \
       all(tablero[i][2-i] == jugador for i in range(3)):
        return True

    return False

# Función para generar todos los posibles movimientos restantes
def generar_movimientos_restantes(tablero, jugador):
    movimientos = []
    for i in range(3):
        for j in range(3):
            if tablero[i][j] == ' ':
                nuevo_tablero = copy.deepcopy(tablero)
                nuevo_tablero[i][j] = jugador
                movimientos.append(nuevo_tablero)
    return movimientos

# Heurística para evaluar el tablero
def evaluar_tablero(tablero):
    if verificar_ganador(tablero, 'X'):
        return 10  # Triunfo de la computadora
    elif verificar_ganador(tablero, 'O'):
        return -10  # Triunfo del humano
    else:
        return 0  # Empate

# Función principal del juego
def jugar_gato():
    tablero = [[' ']*3 for _ in range(3)]
    jugador_actual = 'X'

    while True:
        imprimir_tablero(tablero)

        if jugador_actual == 'X':
            # Turno de la computadora
            movimientos = generar_movimientos_restantes(tablero, jugador_actual)
            puntajes = [evaluar_tablero(movimiento) for movimiento in movimientos]
            mejor_movimiento_idx = puntajes.index(max(puntajes))
            tablero = movimientos[mejor_movimiento_idx]
            if verificar_ganador(tablero, jugador_actual):
                imprimir_tablero(tablero)
                print("¡La computadora ha ganado!")
                break
            jugador_actual = 'O'
        else:
            # Turno del humano
            fila = int(input("Ingrese el número de fila: "))
            columna = int(input("Ingrese el número de columna: "))
            if tablero[fila][columna] != ' ':
                print("¡Casilla ocupada! Inténtalo de nuevo.")
                continue
            tablero[fila][columna] = jugador_actual
            imprimir_tablero(tablero)
            movimientos_humano = generar_movimientos_restantes(tablero, 'X')
            print("Posibles movimientos de la computadora:")
            for i, movimiento in enumerate(movimientos_humano):
                print(f"Opción {i + 1}:")
                imprimir_tablero(movimiento)
            if verificar_ganador(tablero, jugador_actual):
                print("¡Has ganado!")
                break
            jugador_actual = 'X'

# Iniciar el juego
jugar_gato()