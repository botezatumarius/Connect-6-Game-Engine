"""Connect6 board that keeps, for every window of six cells, the stones each colour has in it."""
from defines import Defines

SIZE = Defines.BOARD_SIZE
WIN = Defines.WIN_LENGTH
THREAT = WIN - 2  # Stones in a live window that a single two-stone move completes.
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))  # Horizontal, vertical, diagonal, anti-diagonal.


def opponent(color):
    """Colour of the other player."""
    return Defines.BLACK + Defines.WHITE - color


def build_windows():
    """Every line of WIN consecutive cells on the board (924 on a 19x19 board)."""
    windows = []
    for row in range(SIZE):
        for col in range(SIZE):
            for dr, dc in DIRECTIONS:
                # Keep the window only if its last cell is still on the board.
                if 0 <= row + dr * (WIN - 1) < SIZE and 0 <= col + dc * (WIN - 1) < SIZE:
                    windows.append(tuple((row + dr * i) * SIZE + col + dc * i for i in range(WIN)))
    return tuple(windows)


def build_cell_windows():
    """For each cell, the windows that contain it (at most 24)."""
    cell_windows = [[] for _ in range(SIZE * SIZE)]
    for window, cells in enumerate(WINDOWS):
        for cell in cells:
            cell_windows[cell].append(window)
    return tuple(map(tuple, cell_windows))


WINDOWS = build_windows()
CELL_WINDOWS = build_cell_windows()


class Board:
    """Board state with window statistics updated incrementally on every stone."""

    def __init__(self):
        self.cells = [Defines.NOSTONE] * (SIZE * SIZE)  # Cell index = row * SIZE + col.
        self.stones = 0
        self.winner = Defines.NOSTONE
        self.history = []  # (move, colour) pairs, so moves can be undone.
        # Indexed by colour: stones in each window, live windows (no opponent stone) by
        # stone count, and live windows with at least THREAT stones.
        self.counts = [None, [0] * len(WINDOWS), [0] * len(WINDOWS)]
        self.lines = [None, [0] * (WIN + 1), [0] * (WIN + 1)]
        self.threats = [None, set(), set()]

    def is_full(self):
        """True when no empty cell is left (the game is a draw if nobody has won)."""
        return self.stones == len(self.cells)

    def blockers(self, color):
        """Stones the opponent needs to kill every set `color` can complete next turn (max 3).

        It is the smallest set of empty cells that hits every threat window of `color`.
        """
        gaps = [{cell for cell in WINDOWS[window] if not self.cells[cell]} for window in self.threats[color]]
        if not gaps:
            return 0
        if set.intersection(*gaps):
            return 1  # One cell lies in every threat window.
        for cell in set.union(*gaps):
            rest = [gap for gap in gaps if cell not in gap]
            if set.intersection(*rest):
                return 2  # This cell plus one more cover all the windows.
        return 3  # A two-stone move cannot stop them all.

    def play(self, move, color):
        """Place the one or two stones of `move`."""
        for cell in move:
            self.put(cell, color)
        self.history.append((move, color))

    def undo(self):
        """Take back the last move."""
        move, color = self.history.pop()
        for cell in reversed(move):
            self.remove(cell, color)
        self.winner = Defines.NOSTONE  # Only the last move can have won.

    def put(self, cell, color):
        """Place one stone and update the statistics of the windows through it."""
        other = opponent(color)
        mine, theirs = self.counts[color], self.counts[other]
        self.cells[cell] = color
        self.stones += 1
        for window in CELL_WINDOWS[cell]:
            n, m = mine[window], theirs[window]
            mine[window] = n + 1
            if m == 0:
                # Our live window grows from n to n + 1 stones (index 0 is unused).
                self.lines[color][n] -= 1
                self.lines[color][n + 1] += 1
                if n + 1 >= THREAT:
                    self.threats[color].add(window)
                    if n + 1 == WIN:
                        self.winner = color
            elif n == 0:
                # The opponent's live window now holds our stone: it is dead.
                self.lines[other][m] -= 1
                self.threats[other].discard(window)

    def remove(self, cell, color):
        """Remove one stone, reverting exactly what put() did."""
        other = opponent(color)
        mine, theirs = self.counts[color], self.counts[other]
        self.cells[cell] = Defines.NOSTONE
        self.stones -= 1
        for window in CELL_WINDOWS[cell]:
            n, m = mine[window], theirs[window]
            mine[window] = n - 1
            if m == 0:
                # Our live window shrinks from n to n - 1 stones.
                self.lines[color][n] -= 1
                self.lines[color][n - 1] += 1
                if n == THREAT:
                    self.threats[color].discard(window)
            elif n == 1:
                # Our only stone leaves the window: it is live again for the opponent.
                self.lines[other][m] += 1
                if m >= THREAT:
                    self.threats[other].add(window)
