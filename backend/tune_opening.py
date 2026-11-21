import random

from brain import get_brain
from connect4_env import Connect4
from opponents import MinimaxPlayer

N = 30
CENTER = [2, 3, 4]

# (label, number of AI moves that are random among the center columns)
CONFIGS = [
    ("greedy", 0),
    ("random 1st move", 1),
    ("random 1st+2nd", 2),
]


def play(brain, opp, ai_first, k, rng):
    me = 1 if ai_first else -1
    g = Connect4()
    ai_moves = 0
    moves = []
    while not g.done:
        if g.current_player == me:
            if ai_moves < k:
                legal = [c for c in CENTER if g.is_legal(c)]
                col = rng.choice(legal)
            else:
                col = brain.choose(g)
            ai_moves += 1
        else:
            col = opp(g)
        g.step(col)
        moves.append(col)
    return g.winner == me, tuple(moves)


def run(fname, label, k, depth):
    brain = get_brain(fname)
    rng = random.Random(1)
    wins = 0
    distinct = set()
    for ai_first in (True, False):
        for i in range(N):
            opp = MinimaxPlayer(depth, seed=i).choose
            won, moves = play(brain, opp, ai_first, k, rng)
            wins += won
            distinct.add(moves)
    total = 2 * N
    print(f"{fname:12s} {label:16s} vs minimax{depth} "
          f"{wins}/{total} = {100 * wins / total:.0f}% | distinct {len(distinct)}")


def main():
    for depth in (1, 2):
        print(f"== vs Minimax depth {depth} ==")
        for fname in ("ep_5000.pt", "ep_15000.pt", "final.pt"):
            for label, k in CONFIGS:
                run(fname, label, k, depth)
            print()


if __name__ == "__main__":
    main()
