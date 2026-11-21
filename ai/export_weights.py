import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from connect4_env import Connect4  # noqa: E402
from model import DQN  # noqa: E402

FILES = ["ep_500.pt", "ep_5000.pt", "final.pt"]
NAMES = {
    "fc1.weight": "fc1_w", "fc1.bias": "fc1_b",
    "fc2.weight": "fc2_w", "fc2.bias": "fc2_b",
    "out.weight": "out_w", "out.bias": "out_b",
}
SHAPES = {
    "fc1_w": (128, 84), "fc1_b": (128,),
    "fc2_w": (128, 128), "fc2_b": (128,),
    "out_w": (7, 128), "out_b": (7,),
}
TOLERANCE = 1e-4


def np_forward(w, x):
    """The same network as model.py, in plain NumPy. x has shape [N, 84]."""
    h1 = np.maximum(x @ w["fc1_w"].T + w["fc1_b"], 0.0)
    h2 = np.maximum(h1 @ w["fc2_w"].T + w["fc2_b"], 0.0)
    q = h2 @ w["out_w"].T + w["out_b"]
    return q, h1, h2


def collect_positions(w, n_games, seed):
    """Play games (40% random moves, 60% the network's best move) and keep every position."""
    rng = random.Random(seed)
    states = []
    for _ in range(n_games):
        g = Connect4()
        while not g.done:
            state = np.array(g.get_state(), dtype=np.float32)
            states.append(state)
            legal = list(g.legal_moves())
            if rng.random() < 0.4:
                col = rng.choice(legal)
            else:
                q = np_forward(w, state[None, :])[0][0]
                col = max(legal, key=lambda c: q[c])
            g.step(col)
    return np.stack(states)


def export(fname):
    ckpt = torch.load(os.path.join(CKPT_DIR, fname), map_location="cpu", weights_only=False)
    sd = ckpt["policy"]
    missing = [k for k in NAMES if k not in sd]
    if missing:
        print(f"{fname}: the layer names are different from what I expected.")
        print("Found:", {k: tuple(v.shape) for k, v in sd.items()})
        sys.exit(1)
    w = {NAMES[k]: sd[k].detach().cpu().numpy().astype(np.float32) for k in NAMES}
    for k, shape in SHAPES.items():
        if w[k].shape != shape:
            print(f"{fname}: unexpected shape for {k}: {w[k].shape}, expected {shape}")
            sys.exit(1)
    out = os.path.join(CKPT_DIR, fname.replace(".pt", ".npz"))
    np.savez(out, **w)
    return ckpt, out


def main():
    all_ok = True
    for i, fname in enumerate(FILES):
        ckpt, out = export(fname)

        # Test the saved file itself, not the arrays still in memory.
        loaded = np.load(out)
        w = {k: loaded[k] for k in loaded.files}

        model = DQN()
        model.load_state_dict(ckpt["policy"])
        model.eval()

        states = collect_positions(w, 40, seed=3 + i)
        with torch.no_grad():
            q, h1, h2 = model.forward_with_activations(torch.tensor(states))
        qn, h1n, h2n = np_forward(w, states)

        d_q = float(np.abs(q.numpy() - qn).max())
        d_1 = float(np.abs(h1.numpy() - h1n).max())
        d_2 = float(np.abs(h2.numpy() - h2n).max())
        same = float((q.numpy().argmax(axis=1) == qn.argmax(axis=1)).mean())
        ok = max(d_q, d_1, d_2) < TOLERANCE
        all_ok = all_ok and ok

        print(f"{fname} -> {os.path.basename(out)} ({os.path.getsize(out) / 1024:.0f} KB)")
        print(f"  {len(states)} positions | biggest difference: q {d_q:.2e}, layer 1 {d_1:.2e}, layer 2 {d_2:.2e}")
        print(f"  same best column in {100 * same:.1f}% of positions | {'OK' if ok else 'TOO DIFFERENT'}")

    print()
    print("PASS: NumPy matches PyTorch" if all_ok else "FAIL: do not use these files")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
