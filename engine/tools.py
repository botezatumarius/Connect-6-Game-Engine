import time

from defines import Defines, StoneMove

SYMBOLS = {Defines.NOSTONE: "-", Defines.BLACK: "X", Defines.WHITE: "O"}


def init_board(board):
    last = Defines.GRID_NUM - 1
    for i in range(Defines.GRID_NUM):
        board[i][0] = board[0][i] = board[i][last] = board[last][i] = Defines.BORDER
    for i in Defines.PLAYABLE:
        for j in Defines.PLAYABLE:
            board[i][j] = Defines.NOSTONE


def make_move(board, move, color):
    board[move.positions[0].x][move.positions[0].y] = color
    board[move.positions[1].x][move.positions[1].y] = color


def unmake_move(board, move):
    board[move.positions[0].x][move.positions[0].y] = Defines.NOSTONE
    board[move.positions[1].x][move.positions[1].y] = Defines.NOSTONE


def log_to_file(msg):
    try:
        with open(Defines.LOG_FILE, "a") as file:
            file.write(f"[{time.ctime()}] - {msg}\n")
    except OSError:
        pass


# Coordinates use the GUI's letters A-S: the first letter is the row, the second the column.
def cell_name(x, y):
    return chr(ord("A") + x - 1) + chr(ord("A") + y - 1)


def msg2move(msg):
    """Parse 'XY' (one stone) or 'XYXY' (two stones)."""
    move = StoneMove()
    cells = (msg[:2], msg[2:4] or msg[:2])
    for position, (row, col) in zip(move.positions, cells):
        position.x = ord(row) - ord("A") + 1
        position.y = ord(col) - ord("A") + 1
    return move


def move2msg(move):
    first, second = (cell_name(p.x, p.y) for p in move.positions)
    return first if first == second else first + second


def print_board(board):
    letters = [chr(ord("A") + i - 1) for i in Defines.PLAYABLE]
    print("  " + " ".join(letters))
    for x in Defines.PLAYABLE:
        print(letters[x - 1], " ".join(SYMBOLS[board[x][y]] for y in Defines.PLAYABLE))
