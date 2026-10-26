"""Session 5: impact of the search parameters on decisions, playing strength, speed and memory.

1. Decisions: each configuration plays the first move of every certified position; the prover
   checks it (after an attack the win must still be forced, after a defence it must not).
2. Duels: each configuration plays games against the default engine (in this process, both
   colours, every opening), so only the parameter changes.
3. Memory: peak memory of a search with and without the transposition table.

Usage: python experiments/sweeps.py
"""
import csv
import sys
import time
import tracemalloc
from multiprocessing import Pool

from engines import ROOT
from positions import ATTACKS, DEFENCES, board_of, to_move

sys.path.insert(0, str(ROOT / "engine"))
from board import SIZE, Board, opponent  # noqa: E402
from defines import Defines  # noqa: E402
from search_engine import SearchEngine  # noqa: E402
from threat_search import all_defences_lose, wins_in  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
HAND_TUNED = (1, 4, 16, -1, -4, -16, -128)
DECISION_TIME = 4.0  # Seconds per decision (the GUI's "Low" level).
DUEL_TIME = 1.0      # Seconds per move in the duels.
DUEL_TURNS = 60      # Longer duels are scored as draws.
# (parameter, value, SearchEngine arguments), and the value each parameter has in the engine.
DEFAULTS = {"branching": 10, "transposition table": "on", "quiescence plies": 0, "weights": "evolved"}
CONFIGURATIONS = (
    [("branching", n, {"branching": n}) for n in (5, 10, 20)]
    + [("transposition table", name, {"table": on}) for name, on in (("off", False), ("on", True))]
    + [("quiescence plies", q, {"quiescence": q}) for q in (0, 2, 4)]
    + [("weights", "hand-tuned", {"weights": HAND_TUNED}), ("weights", "evolved", {})]
)
TIME_LIMITS = (0.5, 1, 2, 4, 8)
OPENINGS = ((), ("JK", "KJ"), ("IJ", "KK"), ("HH", "LL"), ("JL", "LJ"), ("IK", "KK"))


def cell(name):
    return (ord(name[0]) - ord("A")) * SIZE + ord(name[1]) - ord("A")


def correct(moves, kind, turns, move):
    """Check a first move with the threat-space prover of positions.py."""
    board = board_of(moves)
    color = to_move(moves)
    board.play(move, color)
    if kind == "attack":
        blockers = board.blockers(color)
        return bool(board.winner) or blockers >= 3 or bool(blockers and all_defences_lose(board, color, turns - 1))
    return not wins_in(board, opponent(color), turns)


def decide(task):
    """Search the first move of one position with one configuration; return a CSV row."""
    parameter, value, arguments, limit, kind, name, moves, turns = task
    engine = SearchEngine(**arguments)
    start = time.perf_counter()
    move, _ = engine.search(board_of(moves), to_move(moves), Defines.MAX_DEPTH, limit)
    seconds = time.perf_counter() - start
    return [parameter, value, kind, name, int(correct(moves, kind, turns, move)), f"{seconds:.3f}",
            engine.m_depth, engine.m_total_nodes, engine.m_pruned_nodes]


def duel(task):
    """One game between a configuration and the default engine; returns the configuration's points."""
    parameter, value, arguments, opening, config_is_black = task
    board = Board()
    color = Defines.BLACK
    board.play((cell("JJ"),), color)
    if opening:
        color = Defines.WHITE
        board.play(tuple(cell(c) for c in opening), color)
    color = opponent(color)
    config_color = Defines.BLACK if config_is_black else Defines.WHITE
    engines = {config_color: SearchEngine(**arguments), opponent(config_color): SearchEngine()}
    for _ in range(DUEL_TURNS):
        move, _ = engines[color].search(board, color, Defines.MAX_DEPTH, DUEL_TIME)
        board.play(move, color)
        if board.winner or board.is_full():
            break
        color = opponent(color)
    points = 1 if not board.winner else 2 if board.winner == config_color else 0
    return parameter, value, points


def memory(table):
    """Peak memory (MiB) and table entries after searching every position to depth 4."""
    engine = SearchEngine(table=table)
    tracemalloc.start()
    for moves, _ in list(ATTACKS.values()) + list(DEFENCES.values()):
        engine.search(board_of(moves), to_move(moves), 4)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / 2 ** 20, len(engine.m_table) if table else 0


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    positions = [("attack", name, moves, turns) for name, (moves, turns) in ATTACKS.items()]
    positions += [("defence", name, moves, turns) for name, (moves, turns) in DEFENCES.items()]
    tasks = [("time limit (s)", limit, {}, limit, *p) for limit in TIME_LIMITS for p in positions]
    tasks += [(parameter, value, arguments, DECISION_TIME, *p)
              for parameter, value, arguments in CONFIGURATIONS for p in positions]
    games = [(parameter, value, arguments, opening, black)
             for parameter, value, arguments in CONFIGURATIONS if DEFAULTS[parameter] != value
             for opening in OPENINGS for black in (True, False)]

    with Pool(4) as pool:  # Four processes: one per physical core, so timed searches are fair.
        rows = pool.map(decide, tasks)
        with open(RESULTS / "sweeps.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["parameter", "value", "kind", "position", "correct", "seconds", "depth", "nodes",
                             "pruned"])
            writer.writerows(rows)
        print("decisions written", flush=True)
        points = {}
        for parameter, value, score in pool.imap_unordered(duel, games):
            points.setdefault((parameter, value), []).append(score)
    with open(RESULTS / "duels.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["parameter", "value", "games", "wins", "draws", "losses", "score"])
        for (parameter, value), scores in points.items():
            writer.writerow([parameter, value, len(scores), scores.count(2), scores.count(1), scores.count(0),
                             f"{sum(scores) / (2 * len(scores)):.3f}"])
    with open(RESULTS / "memory.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["transposition_table", "peak_mib", "entries"])
        for table in (False, True):
            peak, entries = memory(table)
            writer.writerow(["on" if table else "off", f"{peak:.1f}", entries])
    print("done")


if __name__ == "__main__":
    main()
