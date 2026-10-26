"""Session 5 test battery: each engine plays the certified positions against Cloudict.

An attack succeeds if the engine wins within the proven number of turns; a defence succeeds if
Cloudict has not won after the threatened number of turns plus one.

Usage: python experiments/battery.py v1 v2 v3 v4 [--level 4] [--out battery]
"""
import argparse
import csv
from multiprocessing import Pool

from engines import ROOT, spec
from match import play_game
from positions import ATTACKS, DEFENCES, to_move

RESULTS = ROOT / "experiments" / "results"


def run(task):
    """Play one position; return a CSV row with the outcome and our first move's statistics."""
    engine, kind, name, moves, turns, level = task
    ours = to_move(moves)
    # Our turns plus Cloudict's: attacks need `turns` of ours, defences `turns` + 1 of Cloudict's.
    limit = 2 * turns - 1 if kind == "attack" else 2 * (turns + 1)
    players = {ours: spec(engine), 3 - ours: spec("cloudict")}
    game = play_game(players[1], players[2], moves, level, max_turns=limit)
    success = game.winner == ours if kind == "attack" else game.winner != 3 - ours
    first = game.turns[0]
    return [engine, kind, name, turns, int(success), game.reason, len(game.turns),
            f"{first.seconds:.3f}", first.depth, first.nodes, first.pruned]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("engines", nargs="+")
    parser.add_argument("--level", type=int, default=4, help="GUI level: our seconds, Cloudict's depth")
    parser.add_argument("--jobs", type=int, default=4, help="games played in parallel")
    parser.add_argument("--out", default="battery", help="name of the results file")
    args = parser.parse_args()

    tasks = [(engine, kind, name, moves, turns, args.level)
             for engine in args.engines
             for kind, positions in (("attack", ATTACKS), ("defence", DEFENCES))
             for name, (moves, turns) in positions.items()]
    for engine in args.engines:
        spec(engine)  # Check out the versions before the workers start.
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{args.out}.csv", "w", newline="") as file, Pool(args.jobs) as pool:
        writer = csv.writer(file)
        writer.writerow(["engine", "kind", "position", "turns", "success", "result", "turns_played",
                         "first_move_seconds", "depth", "nodes", "pruned"])
        for row in pool.imap(run, tasks):
            writer.writerow(row)
            print(*row[:6], sep=" | ", flush=True)


if __name__ == "__main__":
    main()
