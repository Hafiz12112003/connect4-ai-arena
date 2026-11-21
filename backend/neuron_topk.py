import numpy as np
import torch

from brain import get_brain, SAMPLE_SIZE
from neuron_activity import collect_states, FILES


def count_report(name, acts):
    n_active = (acts > 0).sum(axis=1)
    print(f"  {name}: active neurons per position: avg {n_active.mean():.1f} | "
          f"lowest {n_active.min()} | highest {n_active.max()} | "
          f"positions with at least {SAMPLE_SIZE} active: {100 * (n_active >= SAMPLE_SIZE).mean():.0f}%")


def main():
    for i, fname in enumerate(FILES):
        brain = get_brain(fname)
        states = collect_states(brain, seed=100 + i)
        x = torch.tensor(np.array(states), dtype=torch.float32)
        with torch.no_grad():
            q, h1, h2 = brain.model.forward_with_activations(x)
        h1 = h1.numpy()
        h2 = h2.numpy()
        W = brain.model.out.weight.detach().numpy()  # [7 columns, 128]

        # How much each layer-2 neuron pushes on the 7 outputs, per position.
        contrib = np.abs(h2[:, None, :] * W[None, :, :]).sum(axis=1)  # [N, 128]
        total = np.maximum(contrib.sum(axis=1), 1e-12)

        top = np.argsort(-h2, axis=1)[:, :SAMPLE_SIZE]
        cov_top = np.take_along_axis(contrib, top, axis=1).sum(axis=1) / total
        cov_even = contrib[:, brain.sample_idx].sum(axis=1) / total

        print(f"== {fname}: {len(states)} positions ==")
        count_report("layer 1", h1)
        count_report("layer 2", h2)
        print("  share of the output signal carried by the 12 shown layer-2 neurons:")
        print(f"    12 strongest for each position: avg {100 * cov_top.mean():.0f}% | worst position {100 * cov_top.min():.0f}%")
        print(f"    12 evenly spaced (current):     avg {100 * cov_even.mean():.0f}% | worst position {100 * cov_even.min():.0f}%")
        print()


if __name__ == "__main__":
    main()
