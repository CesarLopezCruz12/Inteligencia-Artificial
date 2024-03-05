import tkinter as tk
import random
from tkinter import messagebox

def imprimir_tablero(tablero):
    for fila in tablero:
        print("|".join(fila))
    print("\n")

def ganador(tablero, jugador):
    for fila in tablero:
        if all(elem == jugador for elem in fila):
            return True
    for columna in range(3):
        if all(tablero[fila][columna] == jugador for fila in range(3)):
            return True
    if all(tablero[i][i] == jugador for i in range(3)):
        return True
    if all(tablero[i][2-i] == jugador for i in range(3)):
        return True
    return False

class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe")

        self.turn = 'X'
        self.board = [[' ']*3 for _ in range(3)]

        self.buttons = []
        for i in range(3):
            row = []
            for j in range(3):
                button = tk.Button(self.root, text='', font=('Arial', 20), width=5, height=2,
                                   command=lambda i=i, j=j: self.click(i, j))
                button.grid(row=i, column=j, sticky='nsew')
                row.append(button)
            self.buttons.append(row)

        self.menu = tk.Menu(root)
        root.config(menu=self.menu)
        game_menu = tk.Menu(self.menu, tearoff=0)
        game_menu.add_command(label="Jugador vs Computadora", command=self.start_player_vs_computer)
        game_menu.add_command(label="Computadora vs Computadora", command=self.start_computer_vs_computer)
        game_menu.add_command(label="Salir", command=root.quit)
        self.menu.add_cascade(label="Juego", menu=game_menu)

        self.root.mainloop()

    def start_player_vs_computer(self):
        self.machine_player = True
        self.reset_game()

    def start_computer_vs_computer(self):
        self.machine_player = False
        self.reset_game()
        self.play_computer_vs_computer()

    def play_computer_vs_computer(self):
        while True:
            self.machine_move()
            self.root.update()  # Actualiza la interfaz gráfica
            if self.check_winner():
                self.end_game()
                return


    def reset_game(self):
        self.turn = 'X'
        self.board = [[' ']*3 for _ in range(3)]
        for row in self.buttons:
            for button in row:
                button.config(text='')
                button.config(bg='SystemButtonFace')

    def click(self, i, j):
        if self.board[i][j] == ' ':
            self.board[i][j] = self.turn
            self.buttons[i][j].config(text=self.turn)
            if self.check_winner():
                self.end_game()
                return
            self.turn = 'O' if self.turn == 'X' else 'X'
            if self.machine_player and self.turn == 'O':
                self.machine_move()

    def check_winner(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != ' ':
                self.highlight_winner(i, 0, i, 1, i, 2)
                return True
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != ' ':
                self.highlight_winner(0, i, 1, i, 2, i)
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            self.highlight_winner(0, 0, 1, 1, 2, 2)
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            self.highlight_winner(0, 2, 1, 1, 2, 0)
            return True
        if all(self.board[i][j] != ' ' for i in range(3) for j in range(3)):
            return True
        return False

    def highlight_winner(self, *positions):
        for i in range(0, len(positions), 2):
            row, col = positions[i], positions[i+1]
            self.buttons[row][col].config(bg='green')

    def machine_move(self):
        available_moves = [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == ' ']
        if available_moves:
            square = random.choice(available_moves)
            self.click(square[0], square[1])

    def end_game(self):
        winner = self.turn
        if winner == 'X':
            winner = 'Jugador'
        elif winner == 'O':
            winner = 'Computadora'
        else:
            winner = 'Empate'

        answer = messagebox.askquestion("Fin del Juego", f"¡{winner} gana!\n\n¿Quieres jugar de nuevo?")
        if answer == 'yes':
            self.reset_game()
            if not self.machine_player or self.turn == 'X':
                self.turn = 'O'
                self.machine_move()
        else:
            self.root.quit()


if __name__ == '__main__':
    root = tk.Tk()
    app = TicTacToeGUI(root)
