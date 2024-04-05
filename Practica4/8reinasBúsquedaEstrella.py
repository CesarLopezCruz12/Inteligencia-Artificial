import heapq

def calcQueens(size):
    board = [-1] * size
    if queens_A_star(board, size):
        print_board(board)
    else:
        print("No solution found for the given size.")

def queens_A_star(board, size):
    open_list = []
    closed_list = set()
    heapq.heappush(open_list, (0, board[:], 0))  # (f_cost, board, current)
    
    while open_list:
        f_cost, current_board, current = heapq.heappop(open_list)
        closed_list.add(tuple(current_board))
        
        if current == size:
            board[:] = current_board
            return True
        
        for i in range(size):
            current_board[current] = i
            if tuple(current_board) not in closed_list and noConflicts(current_board, current):
                g_cost = current + 1
                h_cost = size - current - 1  # Heurística: número de reinas restantes
                f_cost = g_cost + h_cost
                heapq.heappush(open_list, (f_cost, current_board[:], current + 1))
                
    return False

def noConflicts(board, current):
    for i in range(current):
        if board[i] == board[current] or current - i == abs(board[current] - board[i]):
            return False
    return True

def print_board(board):
    size = len(board)
    for row in range(size):
        for col in range(size):
            if board[row] == col:
                print("Q", end=" ")
            else:
                print(".", end=" ")
        print()

# Ejemplo de uso
calcQueens(8)


