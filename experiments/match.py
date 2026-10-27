"""Plays games between engines through the same text protocol as the GUI (session 5).

Like ConnectMore, each turn sends `new xxx`, the whole move history and `next`, and gives up on
an engine that does not answer within the GUI's 30 s timeout.
"""
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIZE = 19
BLACK, WHITE, DRAW = 1, 2, 0
GUI_TIMEOUT = 30.0
INFO = re.compile(r"info time ([\d.]+)(?: depth (\d+))? nodes (\d+)(?: pruned (\d+))?")


@dataclass
class Turn:
    """One move of a game and, for our engine, its search statistics."""
    color: int
    move: str
    seconds: float
    depth: int = 0
    nodes: int = 0
    pruned: int = 0


@dataclass
class Game:
    """Result of a game: winner colour (0 for a draw), the reason and every turn."""
    black: str
    white: str
    winner: int = DRAW
    reason: str = ""
    turns: list = field(default_factory=list)


class Engine:
    """An engine process driven through its stdin/stdout."""

    def __init__(self, command, cwd, level):
        self.process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("name")
        self.name = self.read_until("name ")[0][5:].strip()
        self.send(f"depth {level}")

    def send(self, line):
        """Write one command line to the engine."""
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def read_until(self, prefix):
        """Lines read up to (and including) the first one starting with `prefix`."""
        lines = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("engine closed its output")
            lines.append(line.strip())
            if line.startswith(prefix):
                return lines[-1], lines

    def next_move(self, history):
        """Send the position as the GUI does and return (move, seconds, statistics)."""
        self.send("new xxx")
        for color, move in history:
            self.send(f"{'black' if color == BLACK else 'white'} {move * 2 if len(move) == 2 else move}")
        start = time.perf_counter()
        self.send("next")
        line, lines = self.read_until("move ")
        seconds = time.perf_counter() - start
        stats = next((INFO.match(l) for l in reversed(lines) if INFO.match(l)), None)
        return line[5:].strip().upper(), seconds, stats

    def close(self):
        """Ask the engine to quit, killing it if it does not."""
        try:
            self.send("quit")
            self.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self.process.kill()


def cells_of(move):
    """Board cells (row, col) of a move written as 'XY' or 'XYXY'."""
    return {(ord(move[i]) - ord("A"), ord(move[i + 1]) - ord("A")) for i in range(0, len(move), 2)}


def connects_six(board, row, col):
    """True if the stone at (row, col) is part of six or more in a row."""
    color = board[row][col]
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < SIZE and 0 <= c < SIZE and board[r][c] == color:
                count += 1
                r, c = r + sign * dr, c + sign * dc
        if count >= 6:
            return True
    return False


def play_game(black_spec, white_spec, opening=(), level=4, max_turns=None):
    """Play a game from `opening` (a list of (colour, move)); the arbiter checks every move."""
    engines = {BLACK: Engine(*black_spec, level), WHITE: Engine(*white_spec, level)}
    game = Game(engines[BLACK].name, engines[WHITE].name)
    board = [[DRAW] * SIZE for _ in range(SIZE)]
    history = []
    for color, move in opening:
        for r, c in cells_of(move):
            board[r][c] = color
        history.append((color, move))
    color = WHITE if history and history[-1][0] == BLACK else BLACK
    try:
        while max_turns is None or len(game.turns) < max_turns:
            engine = engines[color]
            move, seconds, stats = engine.next_move(history)
            cells = cells_of(move)
            # Arbiter: a move needs two new stones (one on black's very first move) on empty cells.
            if (len(cells) != (1 if not history else 2) or len(move) not in (2, 4)
                    or any(not (0 <= r < SIZE and 0 <= c < SIZE) or board[r][c] for r, c in cells)):
                game.winner, game.reason = 3 - color, f"illegal move {move} by {engine.name}"
                break
            if seconds > GUI_TIMEOUT:
                game.winner, game.reason = 3 - color, f"timeout of {engine.name}"
                break
            turn = Turn(color, move, seconds)
            if stats:
                turn.depth, turn.nodes, turn.pruned = int(stats[2] or 0), int(stats[3]), int(stats[4] or 0)
            game.turns.append(turn)
            for r, c in cells:
                board[r][c] = color
            history.append((color, move))
            if any(connects_six(board, r, c) for r, c in cells):
                game.winner, game.reason = color, f"{engine.name} connects 6"
                break
            if all(all(row) for row in board):
                game.reason = "full board"
                break
            color = 3 - color
        else:
            game.reason = f"stopped after {max_turns} turns"
    finally:
        for engine in engines.values():
            engine.close()
    return game


if __name__ == "__main__":
    from engines import spec
    result = play_game(spec(sys.argv[1]), spec(sys.argv[2]), level=int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    for turn in result.turns:
        print(f"{'black' if turn.color == BLACK else 'white'} {turn.move:5s} {turn.seconds:6.2f}s "
              f"depth {turn.depth} nodes {turn.nodes}")
    print(result.black, "vs", result.white, "->", result.reason)
