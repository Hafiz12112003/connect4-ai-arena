"""
The DQN brain: a plain fully connected neural network.

    84 inputs -> 128 -> 128 -> 7 Q-values

Input  : 84 numbers (the board, from the view of the player to move).
Output : 7 Q-values, one per column. A Q-value is the AI's guess of
         "how good is it to drop a disc in this column?". Higher = better.

No CNN on purpose: every neuron here is a real number we can show in the
web app's "AI Brain" panel, so the picture is an honest view of the model.
"""
import torch
import torch.nn as nn

INPUT_SIZE = 84
HIDDEN_SIZE = 128
OUTPUT_SIZE = 7


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)    # 84  -> 128
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)   # 128 -> 128
        self.out = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)   # 128 -> 7

    def forward(self, x):
        # ReLU keeps positive values and turns negatives into 0.
        # This is what lets the network learn non-straight-line patterns.
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.out(h2)

    def forward_with_activations(self, x):
        # Same as forward(), but also returns what each hidden layer produced.
        # The web app uses this to draw the real activations of the brain.
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        q = self.out(h2)
        return q, h1, h2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = DQN()
    print(model)

    # A batch of 5 fake boards -> should give 5 rows of 7 Q-values
    x = torch.rand(5, INPUT_SIZE)
    q = model(x)
    assert q.shape == (5, 7), "output shape wrong"
    print("PASS output shape:", tuple(q.shape))

    # One board with activations
    q1, h1, h2 = model.forward_with_activations(x[:1])
    assert h1.shape == (1, 128) and h2.shape == (1, 128), "hidden shape wrong"
    assert torch.allclose(q1, model(x[:1])), "activations path gives different Q-values"
    assert (h1 >= 0).all() and (h2 >= 0).all(), "ReLU outputs should never be negative"
    print("PASS hidden layers are 128 wide, same Q-values both ways")

    total = count_parameters(model)
    assert total == 28295, f"expected 28295 parameters, got {total}"
    print("PASS parameter count:", total)
    print("\nAll tests passed.")
