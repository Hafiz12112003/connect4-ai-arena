"""
Diagnostic: how does the trained AI actually play?

For every AI move in 200 games against each opponent, check two basic skills:
  - missed win   : the AI could win this move but played something else
  - missed block : the opponent threatened to win next move and the AI did not stop it
Run:  python ai/diagnose.py [checkpoint file name, default final.pt]
"""
import os
import random
import sys

from agent import DQNAgent
from connect4_env import Connect4
from opponents import RandomPlayer, MinimaxPlayer
from train import NetPlayer

HERE = os.path.dirname(os.path.abspath(__file__))


def immediate_wins(game, player):
    """Columns where `player` would win right now by dropping a disc there."""
    cols = []
    for c in game.legal_moves():
        g = game.copy()
        g.current_player = player
        g.step(c)
        if g.winner == player:
            cols.append(c)
    return cols


def diagnose(player, opponent, n_games):
    s = dict(w=0, d=0, l=0, win_turns=0, missed_win=0, block_turns=0, missed_block=0)
    for i in range(n_games):
        side = 1 if i % 2 == 0 else -1
        game = Connect4()
        while not game.done:
            if game.current_player == side:
                wins = immediate_wins(game, side)
                threats = immediate_wins(game, -side)
                move = player.choose(game)
                if wins:
                    s["win_turns"] += 1
                    if move not in wins:
                        s["missed_win"] += 1
                elif len(threats) == 1:          # exactly one threat: it can be blocked
                    s["block_turns"] += 1
                    if move not in threats:
                        s["missed_block"] += 1
            else:
                move = opponent.choose(game)
            game.step(move)
        if game.winner == side:
            s["w"] += 1
        elif game.winner == 0:
            s["d"] += 1
        else:
            s["l"] += 1
    return s


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "final.pt"
    path = os.path.join(HERE, "checkpoints", name)
    agent = DQNAgent()
    agent.load(path, training=False)
    me = NetPlayer(agent.policy_net)
    random.seed(1)
    n = 200
    print(f"Checkpoint: {name} | {n} games per opponent\n")
    for label, opp in (("Random", RandomPlayer(seed=1)),
                       ("Minimax depth 1", MinimaxPlayer(depth=1, seed=1)),
                       ("Minimax depth 2", MinimaxPlayer(depth=2, seed=1))):
        s = diagnose(me, opp, n)
        mw = 100 * s["missed_win"] / max(1, s["win_turns"])
        mb = 100 * s["missed_block"] / max(1, s["block_turns"])
        print(f"vs {label:<16} {s['w']}W {s['d']}D {s['l']}L ({100 * s['w'] / n:.0f}% win)")
        print(f"   missed an available win : {s['missed_win']:>3} of {s['win_turns']:>3} chances ({mw:.0f}%)")
        print(f"   failed to block a threat: {s['missed_block']:>3} of {s['block_turns']:>3} chances ({mb:.0f}%)\n")
