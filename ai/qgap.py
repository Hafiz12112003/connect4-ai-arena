"""
Q-gap check: at positions where a basic tactic exists, how does the brain value it?
For "win now" positions: Q of the winning column vs the best other column.
For "must block" positions: Q of the blocking column vs the best other column.
Run:  python ai/qgap.py [checkpoint file name, default final.pt]
"""
import os
import random
import sys

import numpy as np

from agent import DQNAgent
from connect4_env import Connect4
from opponents import RandomPlayer
from train import NetPlayer, immediate_wins

HERE = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1] if len(sys.argv) > 1 else "final.pt"
agent = DQNAgent()
agent.load(os.path.join(HERE, "checkpoints", name), training=False)
mover = NetPlayer(agent.policy_net, epsilon=0.3)   # some randomness for varied positions
opponent = RandomPlayer(seed=1)
random.seed(1)

win_q, win_alt, blk_q, blk_alt = [], [], [], []
for i in range(200):
    side = 1 if i % 2 == 0 else -1
    game = Connect4()
    while not game.done:
        if game.current_player == side:
            q = agent.q_values(game.get_state())
            legal = game.legal_moves()
            wins = immediate_wins(game, side)
            threats = immediate_wins(game, -side)
            if wins:
                others = [c for c in legal if c not in wins]
                if others:
                    win_q.append(max(q[c] for c in wins))
                    win_alt.append(max(q[c] for c in others))
            elif len(threats) == 1:
                others = [c for c in legal if c != threats[0]]
                if others and threats[0] in legal:
                    blk_q.append(q[threats[0]])
                    blk_alt.append(max(q[c] for c in others))
            move = mover.choose(game)
        else:
            move = opponent.choose(game)
        game.step(move)


def report(title, good, alt, ideal):
    good, alt = np.array(good), np.array(alt)
    gap = good - alt
    print(f"{title}  ({len(good)} positions)")
    print(f"   mean Q of the right column   : {good.mean():+.3f}   (ideal about {ideal})")
    print(f"   mean Q of best other column  : {alt.mean():+.3f}")
    print(f"   mean gap (right - other)     : {gap.mean():+.3f}")
    print(f"   right column ranked on top   : {100 * (gap > 0).mean():.0f}% of the time\n")


print(f"Checkpoint: {name}\n")
report("WIN NOW positions", win_q, win_alt, "+1.0")
report("MUST BLOCK positions", blk_q, blk_alt, "close to the best other column or higher")
