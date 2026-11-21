import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_DIR = os.path.join(ROOT, "ai")
CKPT_DIR = os.path.join(AI_DIR, "checkpoints")
if AI_DIR not in sys.path:
    sys.path.insert(0, AI_DIR)

from connect4_env import Connect4  # noqa: E402
from model import DQN  # noqa: E402

# OLD mapping, kept only because backend/tune_variety.py imports it.
# The real difficulty mapping is LEVELS in backend/games.py.
DIFFICULTIES = {
    "easy": "ep_500.pt",
    "medium": "ep_5000.pt",
    "hard": "ep_15000.pt",
    "expert": "final.pt",
}

SAMPLE_SIZE = 12


class Brain:
    """One loaded checkpoint. think() returns a move plus real network data."""

    def __init__(self, filename):
        self.filename = filename
        path = os.path.join(CKPT_DIR, filename)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.ckpt_keys = list(ckpt.keys())
        self.model = DQN()
        self.model.load_state_dict(ckpt["policy"])
        self.model.eval()
        # Old evenly spaced sample. The picture no longer uses it, but
        # neuron_activity.py and neuron_topk.py still do.
        self.sample_idx = np.linspace(0, 127, SAMPLE_SIZE).astype(int).tolist()

    def think(self, game, margin=0.0, rng=None):
        rng = rng or random
        state = game.get_state()
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q, h1, h2 = self.model.forward_with_activations(x)
            q = q.reshape(-1)
            h1 = h1.reshape(-1)
            h2 = h2.reshape(-1)

            # The 12 strongest neurons of each layer for THIS position.
            i1 = torch.topk(h1, SAMPLE_SIZE).indices
            i2 = torch.topk(h2, SAMPLE_SIZE).indices
            h1s = h1[i1]
            h2s = h2[i2]

            w2 = self.model.fc2.weight  # [128 target, 128 source]
            wo = self.model.out.weight  # [7 columns, 128 source]

            # Signal on each line = source activation x trained weight on that line.
            pre = w2[i2][:, i1] * h1s.unsqueeze(0)  # [target j, source i]
            c12 = pre.T                              # [source i][target j]
            c2o = (wo[:, i2] * h2s.unsqueeze(0)).T   # [source i][column]

            # How much of the real signal the drawn neurons carry.
            full_in = (w2[i2] * h1.unsqueeze(0)).abs().sum().item()
            cov_h1 = pre.abs().sum().item() / full_in if full_in > 1e-12 else 0.0
            full_out = (wo * h2.unsqueeze(0)).abs().sum().item()
            cov_out = c2o.abs().sum().item() / full_out if full_out > 1e-12 else 0.0

            active1 = int((h1 > 0).sum().item())
            active2 = int((h2 > 0).sum().item())
        q = q.numpy()

        legal = list(game.legal_moves())
        best_q = max(q[c] for c in legal)
        if margin <= 0:
            column = max(legal, key=lambda c: q[c])
        else:
            near = [c for c in legal if q[c] >= best_q - margin]
            column = rng.choice(near)

        return {
            "column": int(column),
            "q_values": [float(v) for v in q],
            "legal": [int(c) for c in legal],
            "input": [float(v) for v in state],
            "hidden1": h1s.tolist(),
            "hidden2": h2s.tolist(),
            "idx_h1": i1.tolist(),
            "idx_h2": i2.tolist(),
            "contrib_h1_h2": c12.tolist(),
            "contrib_h2_out": c2o.tolist(),
            "active_h1": active1,
            "active_h2": active2,
            "coverage_h1": float(cov_h1),
            "coverage_out": float(cov_out),
        }

    def choose(self, game, margin=0.0, rng=None):
        return self.think(game, margin, rng)["column"]


_cache = {}


def get_brain(filename):
    if filename not in _cache:
        _cache[filename] = Brain(filename)
    return _cache[filename]


def brain_for_difficulty(level):
    return get_brain(DIFFICULTIES[level])


def _self_test():
    print("== Shape checks on final.pt ==")
    brain = get_brain("final.pt")
    print("checkpoint keys:", brain.ckpt_keys)
    g = Connect4()
    for _ in range(6):
        g.step(brain.choose(g))
    info = brain.think(g)
    assert len(info["input"]) == 84
    assert len(info["q_values"]) == 7
    assert len(info["hidden1"]) == SAMPLE_SIZE and len(info["hidden2"]) == SAMPLE_SIZE
    assert len(info["idx_h1"]) == SAMPLE_SIZE and len(info["idx_h2"]) == SAMPLE_SIZE
    assert len(info["contrib_h1_h2"]) == SAMPLE_SIZE
    assert all(len(r) == SAMPLE_SIZE for r in info["contrib_h1_h2"])
    assert len(info["contrib_h2_out"]) == SAMPLE_SIZE
    assert all(len(r) == 7 for r in info["contrib_h2_out"])
    assert info["column"] in info["legal"]
    print("shapes OK")

    print()
    print("== Do the picked neurons and line signals match the real network? ==")
    m = brain.model
    x = torch.tensor(g.get_state(), dtype=torch.float32).unsqueeze(0)
    i1 = torch.tensor(info["idx_h1"], dtype=torch.long)
    i2 = torch.tensor(info["idx_h2"], dtype=torch.long)
    with torch.no_grad():
        q, h1, h2 = m.forward_with_activations(x)
        q = q.reshape(-1)
        h1 = h1.reshape(-1)
        h2 = h2.reshape(-1)
        # shown neurons really are the strongest ones
        rest1 = torch.ones(128, dtype=torch.bool)
        rest1[i1] = False
        rest2 = torch.ones(128, dtype=torch.bool)
        rest2[i2] = False
        top_ok = (h1[i1].min() >= h1[rest1].max()) and (h2[i2].min() >= h2[rest2].max())
        assert bool(top_ok), "the shown neurons are not the strongest"
        # the reported activations are the real ones
        assert torch.allclose(torch.tensor(info["hidden1"]), h1[i1])
        assert torch.allclose(torch.tensor(info["hidden2"]), h2[i2])
        # output layer is a plain linear layer on h2
        q_check = h2 @ m.out.weight.T + m.out.bias
        d1 = (q - q_check).abs().max().item()
        full_o = h2.unsqueeze(0) * m.out.weight
        d2 = (full_o.sum(dim=1) + m.out.bias - q).abs().max().item()
        d3 = (torch.tensor(info["contrib_h2_out"]) - full_o[:, i2].T).abs().max().item()
        # layer 2 is linear + ReLU on h1
        pre = h1.unsqueeze(0) * m.fc2.weight
        h2_check = torch.relu(pre.sum(dim=1) + m.fc2.bias)
        d4 = (h2 - h2_check).abs().max().item()
        d5 = (torch.tensor(info["contrib_h1_h2"]) - pre[i2][:, i1].T).abs().max().item()
    print("shown neurons are the strongest, activations are the real ones: OK")
    print(f"output = linear(h2):            max difference {d1:.2e}")
    print(f"signals into outputs add to q:  max difference {d2:.2e}")
    print(f"12x7 matrix is a true slice:    max difference {d3:.2e}")
    print(f"h2 = relu(linear(h1)):          max difference {d4:.2e}")
    print(f"12x12 matrix is a true slice:   max difference {d5:.2e}")
    for name, d in (("1", d1), ("2", d2), ("3", d3), ("4", d4), ("5", d5)):
        assert d < 1e-4, f"check {name} failed"
    assert 0.0 <= info["coverage_h1"] <= 1.0 and 0.0 <= info["coverage_out"] <= 1.0
    print("all checks passed")

    print()
    print("== Coverage over real positions (30 games each, 25% random moves for variety) ==")
    for fname in ("ep_500.pt", "ep_5000.pt", "final.pt"):
        b = get_brain(fname)
        rng = random.Random(5)
        cov_in, cov_out, act2, lit = [], [], [], []
        for _ in range(30):
            game = Connect4()
            while not game.done:
                t = b.think(game)
                cov_in.append(t["coverage_h1"])
                cov_out.append(t["coverage_out"])
                act2.append(t["active_h2"])
                lit.append(sum(1 for v in t["hidden2"] if v > 0))
                if rng.random() < 0.25:
                    col = rng.choice(list(game.legal_moves()))
                else:
                    col = t["column"]
                game.step(col)
        print(f"{fname:12s} {len(cov_in)} positions | "
              f"layer 1 -> layer 2 signal carried: avg {100 * np.mean(cov_in):.0f}% (lowest {100 * np.min(cov_in):.0f}%) | "
              f"layer 2 -> output signal carried: avg {100 * np.mean(cov_out):.0f}% (lowest {100 * np.min(cov_out):.0f}%) | "
              f"layer-2 neurons lit among the 12 shown: avg {np.mean(lit):.1f}")

    print()
    print("== Full column is never chosen ==")
    f = Connect4()
    for _ in range(6):
        f.step(0)
    assert brain.choose(f) != 0
    print("OK")
    print("Self-test finished.")


if __name__ == "__main__":
    _self_test()
