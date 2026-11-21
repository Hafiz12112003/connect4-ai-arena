"""
The DQN agent: everything about LEARNING lives here. Written by hand in PyTorch.

Parts:
  ReplayMemory - a notebook of past moves the agent learns from
  DQNAgent     - epsilon-greedy choice, target network, Bellman update, illegal-move masking

One transition = (state, action, reward, next_state, next_mask, done), from the point
of view of the player who MOVED (version 2: every move of BOTH players is stored):
  state      = board before the move, seen by the player who moves
  next_state = board right after the move, seen by the NEXT player (the opponent)
  next_mask  = which columns are legal in next_state
  reward     = +1 if this move wins the game, otherwise 0 (a draw is 0)

Zero-sum trick (negamax): a board that is good for the opponent is bad for me, so
  target = reward - gamma * (best Q-value of the next board)
"""
import numpy as np
import torch
import torch.nn.functional as F

from model import DQN, INPUT_SIZE, OUTPUT_SIZE


class ReplayMemory:
    """Fixed-size notebook. When full, the oldest memories are overwritten."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.states = np.zeros((capacity, INPUT_SIZE), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, INPUT_SIZE), dtype=np.float32)
        self.next_masks = np.zeros((capacity, OUTPUT_SIZE), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.index = 0   # where the next memory is written
        self.size = 0    # how many memories we hold

    def push(self, state, action, reward, next_state, next_mask, done):
        i = self.index
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.next_masks[i] = next_mask
        self.dones[i] = float(done)
        self.index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # Random memories (not the latest ones) so the network does not
        # overfit to the game it just played.
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.next_masks[idx]),
            torch.from_numpy(self.dones[idx]),
        )

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.999, batch_size=64, memory_size=50000,
                 target_update=500):
        self.gamma = gamma                    # how much future reward matters (0..1)
        self.epsilon = epsilon                # chance of a random move (exploration)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay    # epsilon is multiplied by this each episode
        self.batch_size = batch_size
        self.target_update = target_update    # copy weights to target net every N learn steps
        self.learn_steps = 0

        self.policy_net = DQN()               # the brain we train and play with
        self.target_net = DQN()               # a slow-moving copy used to compute targets
        self.update_target()
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_size)

    # ---------- choosing moves ----------

    def q_values(self, state):
        """The 7 Q-values for a board (a numpy array of 7 numbers)."""
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.policy_net(x).squeeze(0).numpy()

    def best_action(self, state, legal_mask):
        # Illegal-move masking: full columns get -infinity, so they can never win the argmax.
        q = self.q_values(state)
        q = np.where(legal_mask > 0, q, -np.inf)
        return int(np.argmax(q))

    def select_action(self, state, legal_mask, explore=True):
        """Epsilon-greedy: with chance epsilon play a random LEGAL column, else the best one."""
        if explore and np.random.rand() < self.epsilon:
            return int(np.random.choice(np.flatnonzero(legal_mask)))
        return self.best_action(state, legal_mask)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ---------- learning ----------

    def remember(self, state, action, reward, next_state, next_mask, done):
        self.memory.push(state, action, reward, next_state, next_mask, done)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def learn(self):
        """One learning step on a random batch. Returns the loss, or None if too few memories."""
        if len(self.memory) < self.batch_size:
            return None

        s, a, r, s2, m2, d = self.memory.sample(self.batch_size)

        # What the brain currently predicts for the move that was actually played
        q = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Negamax Bellman target: next_state is seen by the OPPONENT, so what is good
            # for them is bad for me:  target = reward - gamma * (their best Q-value)
            next_q = self.target_net(s2)
            next_q = next_q.masked_fill(m2 == 0, -1e9)    # ignore illegal columns
            best_next = next_q.max(dim=1).values
            target = r - self.gamma * best_next * (1.0 - d)  # game over -> no future

        loss = F.smooth_l1_loss(q, target)    # Huber loss: gentle on big mistakes
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.update_target()
        return float(loss.item())

    # ---------- saving ----------

    def save(self, path, extra=None):
        torch.save({
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "learn_steps": self.learn_steps,
            "extra": extra or {},
        }, path)

    def load(self, path, training=True):
        """training=True resumes training. training=False loads just the brain for playing."""
        ckpt = torch.load(path, map_location="cpu")
        self.policy_net.load_state_dict(ckpt["policy"])
        if training:
            self.target_net.load_state_dict(ckpt["target"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.epsilon = ckpt["epsilon"]
            self.learn_steps = ckpt["learn_steps"]
        else:
            self.update_target()
            self.epsilon = 0.0
        return ckpt["extra"]


def run_tests():
    import os
    import tempfile

    torch.manual_seed(0)
    np.random.seed(0)

    # 1. Illegal-move masking: a full column is never chosen, even if the brain loves it
    agent = DQNAgent()
    agent.epsilon = 0.0
    mask = np.ones(7, dtype=np.float32)
    mask[[2, 5]] = 0
    for _ in range(200):
        state = np.random.rand(84).astype(np.float32)
        assert agent.select_action(state, mask) not in (2, 5), "picked an illegal column"
    with torch.no_grad():
        agent.policy_net.out.bias.zero_()
        agent.policy_net.out.bias[2] = 1000.0     # column 2 looks amazing but is full
    assert agent.best_action(state, mask) != 2, "masking failed"
    print("PASS illegal-move masking")

    # 2. Exploration: random moves are always legal and cover every legal column
    agent.epsilon = 1.0
    picks = {agent.select_action(state, mask) for _ in range(500)}
    assert picks == {0, 1, 3, 4, 6}, f"unexpected explore picks: {picks}"
    print("PASS epsilon-greedy exploration")

    # 3. Learning: a move that always wins (+1, game over) should get Q close to 1
    agent = DQNAgent(batch_size=32, memory_size=1000)
    state = (np.random.rand(84) > 0.7).astype(np.float32)
    zeros = np.zeros(84, dtype=np.float32)
    ones = np.ones(7, dtype=np.float32)
    for _ in range(200):
        agent.remember(state, 3, 1.0, zeros, ones, True)
    for _ in range(600):
        loss = agent.learn()
    q = agent.q_values(state)
    assert abs(q[3] - 1.0) < 0.1, f"Q[3] should be near 1, got {q[3]:.3f}"
    print(f"PASS learning a winning move (Q = {q[3]:.3f})")

    # 4. Negamax Bellman target + masking: the target net says the OPPONENT's best move is worth 0.5
    #    a=0: next board has column 3 legal   -> target = 0 - 0.99 * 0.5 = -0.495
    #    a=1: next board has column 3 FULL    -> their best legal value is 0 -> target = 0
    agent = DQNAgent(batch_size=32, memory_size=1000, target_update=10**9)
    with torch.no_grad():
        agent.target_net.out.weight.zero_()
        agent.target_net.out.bias.zero_()
        agent.target_net.out.bias[3] = 0.5
    no_col3 = np.ones(7, dtype=np.float32)
    no_col3[3] = 0
    for _ in range(100):
        agent.remember(state, 0, 0.0, state, ones, False)
        agent.remember(state, 1, 0.0, state, no_col3, False)
    for _ in range(800):
        agent.learn()
    q = agent.q_values(state)
    assert abs(q[0] - (-0.495)) < 0.1, f"Q[0] should be near -0.495, got {q[0]:.3f}"
    assert abs(q[1] - 0.0) < 0.1, f"Q[1] should be near 0, got {q[1]:.3f}"
    print(f"PASS negamax update + masking (Q0 = {q[0]:.3f}, Q1 = {q[1]:.3f})")

    # 5. Replay memory overwrites the oldest entries when full
    mem = ReplayMemory(10)
    for i in range(25):
        mem.push(state, 0, float(i), state, ones, False)
    assert len(mem) == 10 and mem.rewards.max() == 24.0, "memory wrap failed"
    print("PASS replay memory wraps")

    # 6. Save and load give the same brain
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "test.pt")
        agent.save(path, extra={"episode": 123})
        copy = DQNAgent()
        extra = copy.load(path)
        assert np.allclose(agent.q_values(state), copy.q_values(state)), "loaded brain differs"
        assert extra["episode"] == 123
        player = DQNAgent()
        player.load(path, training=False)
        assert player.epsilon == 0.0 and np.allclose(agent.q_values(state), player.q_values(state))
    print("PASS save and load")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()
