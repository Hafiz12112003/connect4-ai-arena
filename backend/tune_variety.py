import random

from brain import get_brain, DIFFICULTIES
from connect4_env import Connect4
from opponents import MinimaxPlayer

N = 30

# (label, margin, ply_limit): margin applies only while move_count < ply_limit
CONFIGS = [
    ("greedy", 0.0, 0),
    ("margin 0.01", 0.01, 99),
    ("margin 0.02", 0.02, 99),
    ("0.05 first 6 moves only", 0.05, 6),
]


def play(brain, opp, ai_first, margin, ply_limit, rng):
    me = 1 if ai_first else -1
    g = Connect4()
    moves = []
    while not g.done:
        if g.current_player == me:
            m = margin if g.move_count < ply_limit else 0.0
            col = brain.choose(g, m, rng)
        else:
            col = opp(g)
        g.step(col)
        moves.append(col)
    return g.winner == me, tuple(moves)


def run(level, label, margin, ply_limit, opp_name):
    brain = get_brain(DIFFICULTIES[level])
    rng = random.Random(1)
    wins = 0
    distinct = set()
    for ai_first in (True, False):
        for i in range(N):
            if opp_name == "minimax1":
                opp = MinimaxPlayer(1, seed=i).choose
            else:
                r = random.Random(i)
                opp = lambda g, r=r: r.choice(list(g.legal_moves()))
            won, moves = play(brain, opp, ai_first, margin, ply_limit, rng)
            wins += won
            distinct.add(moves)
    total = 2 * N
    print(f"{level:7s} {label:24s} vs {opp_name:8s} "
          f"{wins}/{total} = {100 * wins / total:.0f}% | distinct {len(distinct)}")


def main():
    print("== vs Minimax depth 1 ==")
    for level in ("medium", "hard", "expert"):
        for label, margin, ply in CONFIGS:
            run(level, label, margin, ply, "minimax1")
        print()
    print("== vs Random (is easy a fair beginner level?) ==")
    for level in ("easy", "medium"):
        for label, margin, ply in CONFIGS:
            run(level, label, margin, ply, "random")
        print()


if __name__ == "__main__":
    main()
