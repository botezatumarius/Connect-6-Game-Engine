"""Helpers for the text protocol: move notation, board printing and logging."""
import time

from board import SIZE
from defines import Defines

SYMBOLS = {Defines.NOSTONE: "-", Defines.BLACK: "X", Defines.WHITE: "O"}
LETTERS = [chr(ord("A") + i) for i in range(SIZE)]


def log_to_file(msg):
    """Append a received command to the log file (logging errors are ignored)."""
    try:
        with open(Defines.LOG_FILE, "a") as file:
            file.write(f"[{time.ctime()}] - {msg}\n")
    except OSError:
        pass


# Coordinates use the GUI's letters A-S: the first letter is the row, the second the column.
def msg2move(msg):
    """Parse 'XY' (one stone) or 'XYXY' (two stones) into a tuple of distinct cells."""
    cells = ((ord(msg[i]) - ord("A")) * SIZE + ord(msg[i + 1]) - ord("A") for i in range(0, len(msg) - 1, 2))
    return tuple(dict.fromkeys(cells))  # 'JJJJ' (black's first move) becomes one stone.


def move2msg(move):
    """Write a move in the protocol notation."""
    return "".join(LETTERS[cell // SIZE] + LETTERS[cell % SIZE] for cell in move)


def print_board(board):
    """Print the board, X for black and O for white."""
    print("  " + " ".join(LETTERS))
    for row in range(SIZE):
        print(LETTERS[row], " ".join(SYMBOLS[stone] for stone in board.cells[row * SIZE:(row + 1) * SIZE]))
