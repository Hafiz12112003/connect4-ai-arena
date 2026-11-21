import os
import sys
from collections import Counter

from connect4_env import Connect4
from opponents import AIPlayer, MinimaxPlayer

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def make_ai(name):
    path = os.path.join(CKPT_DIR, name)
    try:
        return AIPlayer(path)
    except Exception:
        return AIPlayer(name)


def play(first, second):
    game = Connect4()
    players = {1: first, -1: second}
    moves = []
    while not game.done:
        col = players[game.current_player].choose(game)
        game.step(col)
        moves.append(col)
    return game.winner, tuple(moves)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "final.pt"
    n = 100
    print(f"Checkpoint: {ckpt}  |  opponent: Minimax depth 2  |  {n} games per colour")
    ai = make_ai(ckpt)
    for ai_first in (True, False):
        ai_color = 1 if ai_first else -1
        results = Counter()
        sequences = set()
        for i in range(n):
            mm = MinimaxPlayer(2, seed=i)
            if ai_first:
                winner, moves = play(ai, mm)
            else:
                winner, moves = play(mm, ai)
            sequences.add(moves)
            if winner == 0:
                results["draw"] += 1
            elif winner == ai_color:
                results["win"] += 1
            else:
                results["loss"] += 1
        label = "AI moves FIRST " if ai_first else "AI moves SECOND"
        print(f"{label}: wins={results['win']} losses={results['loss']} "
              f"draws={results['draw']} | distinct games={len(sequences)} of {n}")


if __name__ == "__main__":
    main()
