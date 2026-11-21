import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_DIR = os.path.join(ROOT, "ai")
CKPT_DIR = os.path.join(AI_DIR, "checkpoints")

SAMPLE_SIZE = 12
KEYS = ("fc1_w", "fc1_b", "fc2_w", "fc2_b", "out_w", "out_b")
SHAPES = {
    "fc1_w": (128, 84), "fc1_b": (128,),
    "fc2_w": (128, 128), "fc2_b": (128,),
    "out_w": (7, 128), "out_b": (7,),
}


def strongest(h, k):
    """Indices of the k largest values, strongest first. Ties go to the lower index."""
    return np.lexsort((np.arange(h.size), -h))[:k]


class Brain:
    """One network loaded from a .npz file. Same think/choose interface as brain.py, NumPy only."""

    def __init__(self, filename):
        self.filename = filename
        base = os.path.splitext(filename)[0]
        data = np.load(os.path.join(CKPT_DIR, base + ".npz"))
        self.w = {k: data[k].astype(np.float32) for k in KEYS}
        for k, shape in SHAPES.items():
            if self.w[k].shape != shape:
                raise ValueError(f"{filename}: {k} has shape {self.w[k].shape}, expected {shape}")

    def forward(self, x):
        w = self.w
        h1 = np.maximum(w["fc1_w"] @ x + w["fc1_b"], 0.0)
        h2 = np.maximum(w["fc2_w"] @ h1 + w["fc2_b"], 0.0)
        q = w["out_w"] @ h2 + w["out_b"]
        return q, h1, h2

    def think(self, game, margin=0.0, rng=None):
        rng = rng or random
        w = self.w
        state = np.asarray(game.get_state(), dtype=np.float32)
        q, h1, h2 = self.forward(state)

        # The 12 strongest neurons of each layer for THIS position.
        i1 = strongest(h1, SAMPLE_SIZE)
        i2 = strongest(h2, SAMPLE_SIZE)
        h1s = h1[i1]
        h2s = h2[i2]

        w2 = w["fc2_w"]  # [128 target, 128 source]
        wo = w["out_w"]  # [7 columns, 128 source]

        # Signal on each line = source activation x trained weight on that line.
        pre = w2[i2][:, i1] * h1s[None, :]       # [target j, source i]
        c12 = pre.T                               # [source i][target j]
        c2o = (wo[:, i2] * h2s[None, :]).T        # [source i][column]

        # How much of the real signal the drawn neurons carry.
        full_in = float(np.abs(w2[i2] * h1[None, :]).sum())
        cov_h1 = float(np.abs(pre).sum()) / full_in if full_in > 1e-12 else 0.0
        full_out = float(np.abs(wo * h2[None, :]).sum())
        cov_out = float(np.abs(c2o).sum()) / full_out if full_out > 1e-12 else 0.0

        legal = [int(c) for c in game.legal_moves()]
        best_q = max(q[c] for c in legal)
        if margin <= 0:
            column = max(legal, key=lambda c: q[c])
        else:
            near = [c for c in legal if q[c] >= best_q - margin]
            column = rng.choice(near)

        return {
            "column": int(column),
            "q_values": q.tolist(),
            "legal": legal,
            "input": state.tolist(),
            "hidden1": h1s.tolist(),
            "hidden2": h2s.tolist(),
            "idx_h1": i1.tolist(),
            "idx_h2": i2.tolist(),
            "contrib_h1_h2": c12.tolist(),
            "contrib_h2_out": c2o.tolist(),
            "active_h1": int((h1 > 0).sum()),
            "active_h2": int((h2 > 0).sum()),
            "coverage_h1": cov_h1,
            "coverage_out": cov_out,
        }

    def choose(self, game, margin=0.0, rng=None):
        return self.think(game, margin, rng)["column"]


_cache = {}


def get_brain(filename):
    """Accepts 'final.pt' or 'final.npz'. Both load final.npz."""
    key = os.path.splitext(filename)[0]
    if key not in _cache:
        _cache[key] = Brain(key + ".npz")
    return _cache[key]


def _self_test():
    print("== brain_np loads without PyTorch ==")
    if AI_DIR not in sys.path:
        sys.path.insert(0, AI_DIR)
    from connect4_env import Connect4

    print("torch not imported so far:", "torch" not in sys.modules)
    assert "torch" not in sys.modules, "torch got imported"

    print()
    print("== Shape and rule checks on final.pt ==")
    brain = get_brain("final.pt")
    g = Connect4()
    for _ in range(6):
        g.step(brain.choose(g))
    info = brain.think(g)
    expected = {
        "column", "q_values", "legal", "input", "hidden1", "hidden2", "idx_h1", "idx_h2",
        "contrib_h1_h2", "contrib_h2_out", "active_h1", "active_h2", "coverage_h1", "coverage_out",
    }
    assert set(info) == expected, set(info) ^ expected
    assert len(info["input"]) == 84 and len(info["q_values"]) == 7
    assert len(info["hidden1"]) == SAMPLE_SIZE and len(info["hidden2"]) == SAMPLE_SIZE
    assert all(len(r) == SAMPLE_SIZE for r in info["contrib_h1_h2"]) and len(info["contrib_h1_h2"]) == SAMPLE_SIZE
    assert all(len(r) == 7 for r in info["contrib_h2_out"]) and len(info["contrib_h2_out"]) == SAMPLE_SIZE
    assert info["column"] in info["legal"]
    full = Connect4()
    for _ in range(6):
        full.step(0)
    assert brain.choose(full) != 0
    print("fields, sizes and the full-column rule: OK")

    t0 = time.perf_counter()
    for _ in range(200):
        brain.think(g)
    ms = (time.perf_counter() - t0) / 200 * 1000
    print(f"average time per AI move on this computer: {ms:.2f} ms (this is not Render's speed)")

    print()
    print("== Same answers as the PyTorch brain? ==")
    try:
        from brain import Brain as TorchBrain
    except ImportError:
        print("PyTorch is not installed here, so this comparison is skipped.")
        print("Self-test finished.")
        return

    ok = True
    for fname in ("ep_500.pt", "ep_5000.pt", "final.pt"):
        nb = get_brain(fname)
        tb = TorchBrain(fname)
        rng = random.Random(11)
        n = strict = col_bad = act_bad = 0
        worst = {"q": 0.0, "hidden": 0.0, "lines": 0.0, "coverage": 0.0}
        idx_bad = 0
        for _ in range(25):
            game = Connect4()
            while not game.done:
                a = nb.think(game)
                b = tb.think(game)
                n += 1
                col_bad += a["column"] != b["column"]
                act_bad += (a["active_h1"], a["active_h2"]) != (b["active_h1"], b["active_h2"])
                worst["q"] = max(worst["q"], float(np.abs(np.array(a["q_values"]) - np.array(b["q_values"])).max()))
                for key in ("hidden1", "hidden2"):
                    worst["hidden"] = max(worst["hidden"], float(np.abs(np.array(a[key]) - np.array(b[key])).max()))
                # When all 12 shown neurons are lit there are no ties, so every pick must match exactly.
                if min(a["hidden1"]) > 0 and min(a["hidden2"]) > 0:
                    strict += 1
                    idx_bad += (a["idx_h1"] != b["idx_h1"]) or (a["idx_h2"] != b["idx_h2"])
                    for key in ("contrib_h1_h2", "contrib_h2_out"):
                        worst["lines"] = max(worst["lines"], float(np.abs(np.array(a[key]) - np.array(b[key])).max()))
                    for key in ("coverage_h1", "coverage_out"):
                        worst["coverage"] = max(worst["coverage"], abs(a[key] - b[key]))
                if rng.random() < 0.25:
                    col = rng.choice(list(game.legal_moves()))
                else:
                    col = a["column"]
                game.step(col)
        good = (col_bad == 0 and act_bad == 0 and idx_bad == 0
                and max(worst.values()) < 1e-4)
        ok = ok and good
        print(f"{fname}: {n} positions, {strict} with all 12 neurons lit in both layers")
        print(f"  different best column: {col_bad} | different active-neuron counts: {act_bad} | different neuron picks: {idx_bad}")
        print("  biggest differences: " + ", ".join(f"{k} {v:.1e}" for k, v in worst.items())
              + (" | OK" if good else " | PROBLEM"))

    print()
    print("PASS: the NumPy brain matches the PyTorch brain" if ok else "FAIL: the two brains differ")
    if not ok:
        sys.exit(1)
    print("Self-test finished.")


if __name__ == "__main__":
    _self_test()
