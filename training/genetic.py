"""Genetic algorithm that tunes the evaluation weights by making engines play each other.

Each individual is a weight vector for the features of evaluation.py. The first population is
spread around the hand-tuned weights, and the fitness of each individual is the share of points
it wins in games against other individuals of its generation. At the end, the best individuals
are validated against the hand-tuned weights. Results are written to training/results/.

Usage: python training/genetic.py [--generations 20] [--population 16] [--seed 1] [--progress]
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from board import Board, SIZE, opponent  # noqa: E402
from defines import Defines  # noqa: E402
from evaluation import FEATURES  # noqa: E402
from search_engine import SearchEngine  # noqa: E402

RESULTS = ROOT / "training" / "results"
DEPTH = 2           # Fixed search depth of training games: fast, deterministic and fair.
MAX_TURNS = 60      # Longer games are scored as draws.
GAMES_PER_INDIVIDUAL = 8
TOURNAMENT_SIZE = 3
ELITISM = 2         # Best individuals copied unchanged into the next generation.
MUTATION_RATE = 0.2  # Probability of mutating each gene.
MUTATION_SIGMA = 0.25  # Relative size of a mutation (log-normal factor).
INIT_SIGMA = 0.35   # Spread of the first population around the hand-tuned weights.
VALIDATED = 3       # Best individuals of the last generation compared on validation games.
HAND_TUNED = (1, 4, 16, -1, -4, -16, -128)  # Session 3 weights: starting point and yardstick.


def cell(name):
    """Cell index of a two-letter coordinate such as 'JJ'."""
    return (ord(name[0]) - ord("A")) * SIZE + ord(name[1]) - ord("A")


# Openings (black's centre stone plus white's reply) so the games of a generation differ.
OPENINGS = [((cell("JJ"),), (cell(a), cell(b))) for a, b in (
    ("JK", "KJ"), ("IK", "KK"), ("KK", "II"), ("JL", "LJ"), ("HJ", "JH"), ("KJ", "KL"),
    ("II", "IK"), ("JI", "JK"), ("IJ", "KJ"), ("HH", "LL"), ("IH", "KL"), ("JH", "LJ"))]


def scaled(weights, sigma):
    """Multiply every weight by a random log-normal factor (signs are kept)."""
    return tuple(w * math.exp(random.gauss(0, sigma)) for w in weights)


def play_game(black_weights, white_weights, opening):
    """Play one training game and return the winner's colour (0 for a draw)."""
    board = Board()
    engines = {Defines.BLACK: SearchEngine(black_weights), Defines.WHITE: SearchEngine(white_weights)}
    color = Defines.BLACK
    for move in opening:
        board.play(move, color)
        color = opponent(color)
    for _ in range(MAX_TURNS):
        move, _ = engines[color].search(board, color, DEPTH)
        board.play(move, color)
        if board.winner:
            return color
        if board.is_full():
            return Defines.NOSTONE
        color = opponent(color)
    return Defines.NOSTONE


def play_pairing(pairing):
    """Worker entry point: (black index, white index, weights, weights, opening) -> result."""
    black, white, black_weights, white_weights, opening = pairing
    return black, white, play_game(black_weights, white_weights, opening)


def fitness(population, pool):
    """Share of the available points (2 per win, 1 per draw) each individual wins."""
    n = len(population)
    pairings = []
    for i in range(n):
        # Each individual meets GAMES_PER_INDIVIDUAL / 2 opponents, once with each colour.
        others = [k for k in range(n) if k != i]
        for j in random.sample(others, min(len(others), GAMES_PER_INDIVIDUAL // 2)):
            opening = random.choice(OPENINGS)
            pairings.append((i, j, population[i], population[j], opening))
    points, games = [0] * n, [0] * n
    for black, white, winner in pool.imap_unordered(play_pairing, pairings):
        games[black] += 1
        games[white] += 1
        if winner == Defines.NOSTONE:
            points[black] += 1
            points[white] += 1
        else:
            points[black if winner == Defines.BLACK else white] += 2
    return [p / (2 * g) for p, g in zip(points, games)]


def tournament(population, scores):
    """Tournament selection: the fittest of TOURNAMENT_SIZE random individuals."""
    contenders = random.sample(range(len(population)), TOURNAMENT_SIZE)
    return population[max(contenders, key=lambda i: scores[i])]


def crossover(a, b):
    """Arithmetic crossover: each gene is a random blend of the parents' genes."""
    return tuple(x + random.random() * (y - x) for x, y in zip(a, b))


def mutate(weights):
    """Gaussian (log-normal) mutation of each gene with probability MUTATION_RATE."""
    return tuple(w * math.exp(random.gauss(0, MUTATION_SIGMA)) if random.random() < MUTATION_RATE else w
                 for w in weights)


def evolve(generations, size, pool, log_file):
    """Run the genetic algorithm and return the last generation, fittest first."""
    log = csv.writer(log_file)
    population = [HAND_TUNED] + [scaled(HAND_TUNED, INIT_SIGMA) for _ in range(size - 1)]
    for generation in range(generations):
        start = time.perf_counter()
        scores = fitness(population, pool)
        ranked = sorted(range(size), key=lambda i: scores[i], reverse=True)
        best = population[ranked[0]]
        log.writerow([generation, f"{scores[ranked[0]]:.3f}", f"{sum(scores) / size:.3f}",
                      *(f"{w:.2f}" for w in best)])
        log_file.flush()  # Keep the results of finished generations if the run is interrupted.
        print(f"generation {generation}: best {scores[ranked[0]]:.2f} mean {sum(scores) / size:.2f} "
              f"({time.perf_counter() - start:.0f}s) {tuple(round(w, 1) for w in best)}", flush=True)
        if generation == generations - 1:
            return [population[i] for i in ranked]
        # Next generation: the elite unchanged, the rest bred from tournament winners.
        elite = [population[i] for i in ranked[:ELITISM]]
        children = [mutate(crossover(tournament(population, scores), tournament(population, scores)))
                    for _ in range(size - ELITISM)]
        population = elite + children


def validate(candidates, pool):
    """Score each candidate against the hand-tuned weights on every opening, with both colours."""
    pairings = [(i, -1, weights, HAND_TUNED, opening) for i, weights in enumerate(candidates) for opening in OPENINGS]
    pairings += [(-1, i, HAND_TUNED, weights, opening) for i, weights in enumerate(candidates) for opening in OPENINGS]
    points = [0] * len(candidates)
    for black, white, winner in pool.imap_unordered(play_pairing, pairings):
        candidate = max(black, white)  # The hand-tuned side has index -1.
        if winner == Defines.NOSTONE:
            points[candidate] += 1
        elif (winner == Defines.BLACK) == (black == candidate):
            points[candidate] += 2
    return [p / (4 * len(OPENINGS)) for p in points]


def progress(pool):
    """Score the best individual of every generation against the hand-tuned weights.

    Fitness inside a generation is relative (individuals only meet each other), so this replays
    each generation's best on the validation games to measure the progress in absolute terms.
    """
    with open(RESULTS / "evolution.csv", newline="") as file:
        rows = list(csv.DictReader(file))
    champions = [tuple(float(row[feature]) for feature in FEATURES) for row in rows]
    scores = validate(champions, pool)
    with open(RESULTS / "progress.csv", "w", newline="") as file:
        log = csv.writer(file)
        log.writerow(["generation", "score_vs_hand_tuned"])
        for row, score in zip(rows, scores):
            log.writerow([row["generation"], f"{score:.3f}"])
            print(f"generation {row['generation']}: {score:.3f} against the hand-tuned weights")


def main():
    """Evolve and validate the weights (or only measure the progress with --progress)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--progress", action="store_true", help="only measure each generation's best")
    args = parser.parse_args()
    random.seed(args.seed)
    if args.progress:
        with Pool() as pool:
            progress(pool)
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "evolution.csv", "w", newline="") as file, Pool() as pool:
        csv.writer(file).writerow(["generation", "best_fitness", "mean_fitness", *FEATURES])
        final = evolve(args.generations, args.population, pool, file)
        # The fitness of one generation is noisy: pick the best of the top 3 on validation games.
        candidates = final[:VALIDATED]
        scores = validate(candidates, pool)

    with open(RESULTS / "validation.csv", "w", newline="") as file:
        log = csv.writer(file)
        log.writerow(["candidate", "score_vs_hand_tuned", *FEATURES])
        for i, (weights, score) in enumerate(zip(candidates, scores)):
            log.writerow([i, f"{score:.3f}", *(f"{w:.2f}" for w in weights)])
    best = candidates[max(range(len(candidates)), key=lambda i: scores[i])]
    (RESULTS / "best_weights.json").write_text(json.dumps(dict(zip(FEATURES, best)), indent=2))
    print("validation scores vs hand-tuned:", [round(x, 3) for x in scores])
    print("best weights:", tuple(round(w, 2) for w in best))


if __name__ == "__main__":
    main()
