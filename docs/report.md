# Practice 1: Connect 6 Game Engine — Project Report

**Course:** Techniques in Artificial Intelligence, MSc in Artificial Intelligence, University of Alicante  
**Author:** Marius Botezatu (practice group 1, worked alone)  
**Engine name:** `mbo`  
**Date:** 8 November 2026  
**Repository:** <https://github.com/botezatumarius/Connect-6-Game-Engine>

## 1. Introduction

Connect6 is a two-player game played on a 19×19 Go board. Black places one stone on the first
move; after that, White and Black alternately place **two** stones per turn. The first player
with six or more stones in a row (horizontally, vertically or diagonally) wins, and the game is
a draw if the board fills up first.

The goal of this practice is a game engine that plays Connect6 with both colours from any
position, answers within a time limit, and communicates with the course GUI (ConnectMore)
through its text protocol. It must use the adversarial search algorithms of Unit 3 with
optimisations that make it work in real time, an evaluation function whose weights are tuned
with machine learning, and an experimental study of the effect of each decision.

The work followed the five sessions of the practice guide. The engine at the end of each
session was kept as a tagged version in Git, so that every stage could be measured in the
experiments:

| Version | Session | Main content |
|---|---|---|
| v1 | 1 | End-of-game detection, naive move proposal and evaluation, fixed-depth Minimax, protocol |
| v2 | 2 | Alpha-beta pruning, incremental board, threat-ranked move proposal, enhanced evaluation |
| v3 | 3 | Detection of imminent threats, weighted feature evaluation with hand-tuned weights |
| v4 | 4 | Weights evolved by a genetic algorithm, iterative deepening under a time limit, forward pruning, transposition table, quiescence search, opening book |
| v5 | 5 | Threat-space search, added after the experiments revealed a weakness (final version) |

Section 2 reviews the state of the art, Section 3 explains and justifies the development,
Section 4 presents the experiments of session 5, and Section 5 concludes.

## 2. State of the art

### 2.1 Connect6 and its complexity

Connect6 was introduced by I-Chen Wu in 2005 as a member of the family of *k-in-a-row* games
Connect(m, n, k, p, q): an m×n board, k stones in a row to win, p stones per move and q stones on
the first move [1, 2]. Connect6 is Connect(19, 19, 6, 2, 1). Wu showed why the two-stone rule
makes the game fair: after every turn, the player who has just moved has exactly one stone more
than the opponent, so neither side keeps a permanent advantage of tempo, unlike Gomoku, where the
first player is known to win [3].

The game is very large. Each of the 361 intersections is empty, black or white, so there are at
most 3³⁶¹ ≈ 10¹⁷² states, the same order as Go. The branching factor is far larger than Go's:
with about 300 empty cells, a two-stone move has about 300·299/2 ≈ 45,000 choices. A full-width
search is therefore impossible, and every successful program restricts the moves it considers.

### 2.2 Threats: the central concept

Strong Connect6 programs reason in terms of **threats**. Wu and Huang [2] define them precisely:
a player has *t* threats when the opponent needs at least *t* stones to stop the player from
winning on the next move. Because a player places two stones per turn, a window of six cells
that holds four or five stones of one player and no opposing stone can be completed in one
move. Consequences used throughout this engine:

- One threat (or two) can be answered: the defender uses one (or both) of its stones to block.
- **Three or more threats cannot be answered with two stones: the attacker wins.**
- A move creating two threats forces the opponent to spend its whole turn defending, so the
  attacker keeps the initiative. Chaining such moves until three threats appear is the basis of
  *threat-space search* [3, 4], called "victory by continuous double threat" in Connect6.

Allis et al. [3] introduced threat-space search to solve Gomoku. Wu and Lin [4] adapted proof
search to Connect6 with *relevance zones*, the regions where the defender's stones can affect an
attack. The ConnectMore GUI used in this practice comes with Cloudict [5], an alpha-beta engine
with a "victory by continuous four" (VCF) threat search, which serves as our reference opponent.

### 2.3 Evaluation of Connect6 positions

Since the search cannot reach the end of the game, positions are scored statically. The usual
approach counts **windows** (or *lines*): every segment of six consecutive cells is a possible
winning line. A window containing stones of only one player is *live* for that player, and a
window with stones of both players is *dead*. Programs weight live windows by the number of
stones they contain, often exponentially, because a window with more stones is closer to six;
names such as "live three" or "dead four" describe the same idea for groups of stones. The guide
of this practice asks for the same two aspects: the size of each live set and the number of
stones the opponent needs to kill it.

The weights of such a linear evaluation function, Σ wᵢ·fᵢ(s), are difficult to set by hand
(Unit 3.3). Machine learning can set them instead: evolutionary methods [6] (Unit 4), regression
on game outcomes, or, in recent programs, neural networks trained by reinforcement learning.

### 2.4 Search algorithms and optimisations

Minimax and its pruned version, alpha-beta [7], are the classical algorithms for two-player,
zero-sum games with perfect information (Unit 3.2). Alpha-beta returns the Minimax value while
examining far fewer nodes; its efficiency depends on the order in which moves are tried. The
optimisations of Unit 3.3 address the real-time constraint of game engines:

- **Iterative deepening:** searching depth 1, 2, 3… gives a move whenever time runs out, and each
  iteration orders the next one.
- **Forward pruning (N-best selective search):** only the N most promising moves, ranked by the
  static evaluation, are searched. This is risky but essential with a branching factor of 45,000.
- **Transposition tables with Zobrist hashing** [8]: the same position, reached by different move
  orders, is searched once. Connect6 has many transpositions, since the two stones of a move,
  and the moves of a sequence, can be placed in any order.
- **Quiescence search:** positions are evaluated only when they are quiet, which avoids the
  horizon effect.
- **History heuristic** [9] and **opening books.**

Monte-Carlo Tree Search [10] has also been applied to k-in-a-row games, but random playouts are
weak at the precise tactics (threats) that decide Connect6, so the strongest programs combine
search with threat-based knowledge.

