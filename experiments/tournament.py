"""Session 5 tournament: round robin between engine versions and Cloudict from starting positions.

Every pair of engines plays each opening twice, swapping colours. Writes the games and the
statistics of every move (decision time, depth, expanded and pruned nodes).

Usage: python experiments/tournament.py v1 v2 v3 v4 cloudict [--level 4] [--openings 2] [--out tournament]
"""
import argparse
import csv
from itertools import combinations
from multiprocessing import Pool

from engines import ROOT, spec
from match import BLACK, WHITE, play_game

RESULTS = ROOT / "experiments" / "results"
# Starting positions: the empty board and black's centre stone with three usual white replies.
OPENINGS = {
    "empty": [],
    "JJ/JKKJ": [(BLACK, "JJ"), (WHITE, "JKKJ")],
    "JJ/IJKK": [(BLACK, "JJ"), (WHITE, "IJKK")],
    "JJ/HHLL": [(BLACK, "JJ"), (WHITE, "HHLL")],
}


def run(task):
    """Play one game; return its row and the rows of its moves."""
    black, white, opening, level = task
    game = play_game(spec(black), spec(white), OPENINGS[opening], level)
    winner = {BLACK: black, WHITE: white}.get(game.winner, "draw")
    row = [black, white, opening, winner, game.reason, len(game.turns)]
    moves = [[black if t.color == BLACK else white, opening, i, f"{t.seconds:.3f}", t.depth, t.nodes, t.pruned]
             for i, t in enumerate(game.turns)]
    return row, moves


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("engines", nargs="+")
    parser.add_argument("--level", type=int, default=4, help="GUI level: our seconds, Cloudict's depth")
    parser.add_argument("--openings", type=int, default=2, help="number of starting positions used")
    parser.add_argument("--jobs", type=int, default=4, help="games played in parallel")
    parser.add_argument("--out", default="tournament", help="prefix of the results files")
    args = parser.parse_args()

    openings = list(OPENINGS)[:args.openings]
    tasks = [(black, white, opening, args.level)
             for a, b in combinations(args.engines, 2) for opening in openings for black, white in ((a, b), (b, a))]
    for engine in args.engines:
        spec(engine)  # Check out the versions before the workers start.
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{args.out}_games.csv", "w", newline="") as games_file, \
            open(RESULTS / f"{args.out}_moves.csv", "w", newline="") as moves_file, Pool(args.jobs) as pool:
        games, moves = csv.writer(games_file), csv.writer(moves_file)
        games.writerow(["black", "white", "opening", "winner", "reason", "turns"])
        moves.writerow(["engine", "opening", "turn", "seconds", "depth", "nodes", "pruned"])
        for row, move_rows in pool.imap_unordered(run, tasks):
            games.writerow(row)
            moves.writerows(move_rows)
            print(*row, sep=" | ", flush=True)


if __name__ == "__main__":
    main()
