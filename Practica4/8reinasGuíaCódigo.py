def calcQueens(size):
    board = [-1] * size
    if queens(board, 0, size):
        print_board(board)
    else:
        print("No solution found for the given size.")

def queens(board, current, size):
    if current == size:
        return True
    else:
        for i in range(size):
            board[current] = i
            if noConflicts(board, current):
                if queens(board, current + 1, size):
                    return True
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
