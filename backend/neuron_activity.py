import numpy as np
import torch

import random

from brain import get_brain, SAMPLE_SIZE
from connect4_env import Connect4

FILES = ["ep_500.pt", "ep_5000.pt", "final.pt"]
GAMES = 60


def collect_states(brain, seed):
    rng = random.Random(seed)
    states = []
    for _ in range(GAMES):
        g = Connect4()
        while not g.done:
            states.append(g.get_state())
            if rng.random() < 0.25:
                col = rng.choice(list(g.legal_moves()))
            else:
                col = brain.choose(g)
            g.step(col)
    return states


def layer_report(name, acts, even_idx):
    active = acts > 0
    rate = active.mean(axis=0)
    mean_act = acts.mean(axis=0)
    dead = int((rate == 0).sum())
    busy = int((rate >= 0.5).sum())
    top = np.argsort(-mean_act)[:SAMPLE_SIZE]
    even_ever = int((rate[even_idx] > 0).sum())
    even_avg = float(active[:, even_idx].sum(axis=1).mean())
    top_avg = float(active[:, top].sum(axis=1).mean())
    print(f"  {name}: never active {dead}/128 | active in at least half of positions {busy}/128")
    print(f"    evenly spaced 12 (current): ever active {even_ever}/12 | "
          f"avg active per position {even_avg:.1f}")
    print(f"    12 busiest by mean activation: avg active per position {top_avg:.1f} | "
          f"individual rates {rate[top].min():.2f} to {rate[top].max():.2f}")


def main():
    for i, fname in enumerate(FILES):
        brain = get_brain(fname)
        states = collect_states(brain, seed=100 + i)
        x = torch.tensor(np.array(states), dtype=torch.float32)
        with torch.no_grad():
            q, h1, h2 = brain.model.forward_with_activations(x)
        h1 = h1.numpy()
        h2 = h2.numpy()
        print(f"== {fname}: {len(states)} positions ==")
        layer_report("layer 1", h1, brain.sample_idx)
        layer_report("layer 2", h2, brain.sample_idx)
        print()


if __name__ == "__main__":
    main()
