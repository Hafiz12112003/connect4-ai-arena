"""
Training script for the Connect Four DQN agent (version 5).

Each episode = one full game. The agent plays as player 1 OR -1 (random). The opponent
mix changes as training goes on (curriculum). Every move of BOTH players is stored as a
memory (Q-learning is off-policy, so the agent can learn from moves it did not choose).

NEW in this version:
  - winning moves are stored 3 times (they are rare but very important)
  - the learning rate decays from 5e-4 to 5e-5 over the run (less wobble late on)
  - best.pt: the brain with the best test score so far is kept
  - at the end, best.pt and the last brain play 100 fresh games against each of
    Random / Minimax 1 / Minimax 2, and the better one is saved as final.pt

Opponent types:
    random    - any legal column
    tactical  - takes an immediate win, blocks an immediate threat, otherwise random
                (plays a fully random move 15% of the time)
    mm1n      - Minimax depth 1 that plays a random move 30% of the time
    mm2n      - Minimax depth 2 that plays a random move 30% of the time
    mm2       - plain Minimax depth 2
    past_self - a frozen older copy of the agent

Run:  python ai/train.py
Quick test:  python ai/train.py --episodes 200 --eval-every 100 --eval-games 10 --out-dir ai/smoke_test
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from agent import DQNAgent
from connect4_env import Connect4, ROWS, COLS
from model import DQN
from opponents import RandomPlayer, MinimaxPlayer, play_game

HERE = os.path.dirname(os.path.abspath(__file__))
ZERO_STATE = np.zeros(ROWS * COLS * 2, dtype=np.float32)
ONES_MASK = np.ones(COLS, dtype=np.float32)
LEARN_START = 1000      # start learning once the notebook has this many memories
POOL_SIZE = 12          # how many past versions of the agent we keep
GAMMA = 0.95            # a faster win is worth more than a slower one
LR_START = 5e-4
LR_END = 5e-5
WIN_COPIES = 3          # winning moves are stored this many times
FINAL_CHECK_GAMES = 100

# Curriculum: (training progress below which this mix is used, opponent weights)
STAGES = [
    (0.30, {"tactical": 0.45, "random": 0.25, "mm1n": 0.30}),
    (0.70, {"tactical": 0.20, "random": 0.10, "mm1n": 0.20, "mm2n": 0.20, "past_self": 0.30}),
    (1.01, {"tactical": 0.10, "mm1n": 0.10, "mm2n": 0.25, "mm2": 0.25, "past_self": 0.30}),
]


def opponent_mix(progress):
    for limit, mix in STAGES:
        if progress < limit:
            return mix
    return STAGES[-1][1]


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


class TacticalPlayer:
    """Takes a win if it can, blocks a loss if it must, otherwise plays randomly."""

    def __init__(self, noise=0.0, seed=None):
        self.name = "Tactical"
        self.noise = noise
        self.rng = random.Random(seed)

    def choose(self, game):
        legal = game.legal_moves()
        if self.rng.random() < self.noise:
            return self.rng.choice(legal)
        me = game.current_player
        wins = immediate_wins(game, me)
        if wins:
            return self.rng.choice(wins)
        threats = immediate_wins(game, -me)
        if threats:
            return self.rng.choice(threats)
        return self.rng.choice(legal)


class NoisyPlayer:
    """Wraps another player: with chance `noise` it plays a random legal move instead."""

    def __init__(self, base, noise, seed=None):
        self.base = base
        self.noise = noise
        self.name = f"{base.name} (noisy)"
        self.rng = random.Random(seed)

    def choose(self, game):
        if self.rng.random() < self.noise:
            return self.rng.choice(game.legal_moves())
        return self.base.choose(game)


class NetPlayer:
    """Wraps a neural network as a player (same choose(game) as the other players)."""

    def __init__(self, net, epsilon=0.0, name="AI"):
        self.net = net
        self.epsilon = epsilon
        self.name = name

    def choose(self, game):
        if self.epsilon > 0 and random.random() < self.epsilon:
            return random.choice(game.legal_moves())
        with torch.no_grad():
            x = torch.as_tensor(game.get_state()).unsqueeze(0)
            q = self.net(x).squeeze(0).numpy()
        q = np.where(game.legal_mask() > 0, q, -np.inf)   # never pick a full column
        return int(np.argmax(q))


# ---------- mirror trick ----------
# Connect Four is symmetric: flip the board left-to-right and it is the same game.
# So every memory gives us a second memory for free (column c becomes column 6 - c).

def mirror_state(state):
    return state.reshape(2, ROWS, COLS)[:, :, ::-1].reshape(-1).copy()


def mirror_transition(t):
    s, a, r, s2, m2, d = t
    return (mirror_state(s), COLS - 1 - a, r, mirror_state(s2), m2[::-1].copy(), d)


def check_mirror():
    g1, g2 = Connect4(), Connect4()
    for m in [3, 3, 2, 0]:
        g1.step(m)
    for m in [3, 3, 4, 6]:   # same moves, mirrored
        g2.step(m)
    assert np.array_equal(mirror_state(g1.get_state()), g2.get_state()), "mirror check failed"
    s = np.random.rand(84).astype(np.float32)
    assert np.array_equal(mirror_state(mirror_state(s)), s), "double mirror should be identity"
    print("PASS mirror augmentation")


def check_tactical():
    g = Connect4()
    for m in [0, 6, 1, 6, 2, 6]:    # X has 0,1,2 on the bottom row, X to move
        g.step(m)
    assert TacticalPlayer(seed=1).choose(g) == 3, "tactical player missed a win"
    g = Connect4()
    for m in [6, 0, 6, 1, 5, 2]:    # O has 0,1,2 on the bottom row, X to move
        g.step(m)
    assert TacticalPlayer(seed=1).choose(g) == 3, "tactical player missed a block"
    print("PASS tactical player")


# ---------- one episode ----------

def play_episode(agent, opponent, agent_side):
    """Play one full game and record EVERY move (both players).
    Returns (transitions, result) with result +1 agent won, 0 draw, -1 agent lost.

    A transition = (state, action, reward, next_state, next_mask, done) where
      state      = board before the move, seen by the player who moves
      next_state = board after the move, seen by the NEXT player
      reward     = +1 if this move won the game, else 0
    """
    game = Connect4()
    transitions = []
    while not game.done:
        state = game.get_state()
        mask = game.legal_mask()
        if game.current_player == agent_side:
            action = agent.select_action(state, mask)
        else:
            action = opponent.choose(game)
        game.step(action)
        if game.done:
            reward = 1.0 if game.winner != 0 else 0.0
            transitions.append((state, action, reward, ZERO_STATE, ONES_MASK, True))
        else:
            transitions.append((state, action, 0.0, game.get_state(), game.legal_mask(), False))
    return transitions, int(game.winner * agent_side)


def check_episode():
    agent = DQNAgent()
    agent.epsilon = 1.0
    opp = RandomPlayer(seed=3)
    for i in range(30):
        transitions, result = play_episode(agent, opp, 1 if i % 2 == 0 else -1)
        dones = [t for t in transitions if t[5]]
        assert len(dones) == 1 and transitions[-1][5], "exactly the last move should end the game"
        wins = [t for t in transitions if t[2] == 1.0]
        assert len(wins) == (0 if result == 0 else 1), "one winning move per decided game"
        for s, a, r, s2, m2, d in transitions:
            if not d:
                assert abs(s2.sum() - (s.sum() + 1)) < 1e-6, "one more disc after each move"
    print("PASS episode recording (both players' moves)")


# ---------- test matches ----------

def score_matches(player, opponent, n_games):
    """`player` plays n_games against `opponent`, half as first player, half as second."""
    wins = draws = losses = 0
    for i in range(n_games):
        if i % 2 == 0:
            result = play_game(player, opponent)
        else:
            result = -play_game(opponent, player)
        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1
    return wins, draws, losses


def confirm_score(net, n_games, seed):
    """Average score (win = 1, draw = 0.5) against Random, Minimax 1 and Minimax 2."""
    me = NetPlayer(net)
    total = 0.0
    for opp in (RandomPlayer(seed=seed), MinimaxPlayer(depth=1, seed=seed),
                MinimaxPlayer(depth=2, seed=seed)):
        w, d, l = score_matches(me, opp, n_games)
        total += (w + 0.5 * d) / n_games
    return total / 3


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)       # swap in one step so the dashboard never reads a half-written file


# ---------- main training loop ----------

def train(args):
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.json")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    check_mirror()
    check_tactical()
    check_episode()

    # epsilon falls from 1.0 to 0.05 by 40% of training, then stays at 0.05
    decay = 0.05 ** (1.0 / (0.4 * args.episodes))
    agent = DQNAgent(lr=LR_START, gamma=GAMMA, epsilon=1.0, epsilon_min=0.05,
                     epsilon_decay=decay, batch_size=64, memory_size=100000,
                     target_update=500)

    save_at = sorted({int(x) for x in args.save_at.split(",") if x.strip()})
    sparring = {
        "random": RandomPlayer(seed=args.seed),
        "tactical": TacticalPlayer(noise=0.15, seed=args.seed),
        "mm1n": NoisyPlayer(MinimaxPlayer(depth=1, seed=args.seed), 0.30, seed=args.seed),
        "mm2n": NoisyPlayer(MinimaxPlayer(depth=2, seed=args.seed + 1), 0.30, seed=args.seed + 1),
        "mm2": MinimaxPlayer(depth=2, seed=args.seed + 2),
    }

    pool = []   # frozen copies of the agent's past brains

    def snapshot():
        net = DQN()
        net.load_state_dict(agent.policy_net.state_dict())
        net.eval()
        pool.append(net)
        if len(pool) > POOL_SIZE:
            pool.pop(0)

    def pick_opponent(progress):
        mix = opponent_mix(progress)
        kind = random.choices(list(mix.keys()), weights=list(mix.values()))[0]
        if kind == "past_self":
            return NetPlayer(random.choice(pool), epsilon=0.1)   # a past self, with a little randomness
        return sparring[kind]

    metrics = {
        "config": {
            "episodes": args.episodes, "eval_every": args.eval_every,
            "eval_games": args.eval_games,
            "eval_opponents": ["random", "minimax1", "minimax2"],
            "network": "84-128-128-7", "learning_rate": [LR_START, LR_END], "gamma": GAMMA,
            "batch_size": 64, "mirror_augmentation": True,
            "learns_from_both_players": True, "winning_move_copies": WIN_COPIES,
            "curriculum": [{"until_progress": limit, "mix": mix} for limit, mix in STAGES],
        },
        "history": [],
        "checkpoints": [],
        "finished": False,
    }

    start = time.time()
    losses = []
    best = {"score": -1.0, "episode": None}

    def save_checkpoint(ep, file, label):
        agent.save(os.path.join(out_dir, file), extra={"episode": ep})
        metrics["checkpoints"].append({"episode": ep, "file": file, "label": label})

    def run_eval(ep):
        me = NetPlayer(agent.policy_net)     # greedy: no exploring during a test match
        n = args.eval_games
        avg_loss = float(np.mean(losses)) if losses else None
        losses.clear()
        entry = {"episode": ep}
        results = {}
        score = 0.0
        for key, opp in (("random", RandomPlayer(seed=ep)),
                         ("minimax1", MinimaxPlayer(depth=1, seed=ep)),
                         ("minimax2", MinimaxPlayer(depth=2, seed=ep))):
            w, d, l = score_matches(me, opp, n)
            entry[f"{key}_win_rate"] = w / n
            entry[f"{key}_draw_rate"] = d / n
            entry[f"{key}_loss_rate"] = l / n
            results[key] = 100 * w / n
            score += (w + 0.5 * d) / n
        score /= 3
        entry["loss"] = avg_loss
        entry["epsilon"] = round(agent.epsilon, 4)
        entry["score"] = round(score, 4)
        metrics["history"].append(entry)

        star = ""
        if ep > 0 and score > best["score"]:
            best["score"], best["episode"] = score, ep
            agent.save(os.path.join(out_dir, "best.pt"), extra={"episode": ep})
            star = "  <- new best"
        write_json(metrics_path, metrics)

        loss_text = f"{avg_loss:.4f}" if avg_loss is not None else " n/a  "
        print(f"ep {ep:>6} | eps {agent.epsilon:.3f} | loss {loss_text} | win% vs "
              f"random {results['random']:>3.0f} | minimax1 {results['minimax1']:>3.0f} | "
              f"minimax2 {results['minimax2']:>3.0f} | {(time.time() - start) / 60:.1f} min{star}")

    print(f"Training {args.episodes} episodes. Files go to: {out_dir}")
    snapshot()
    if 0 in save_at:
        save_checkpoint(0, "ep_0.pt", "episode 0")
    run_eval(0)

    for ep in range(1, args.episodes + 1):
        # learning rate falls in a straight line from LR_START to LR_END
        lr = LR_START + (LR_END - LR_START) * (ep / args.episodes)
        for group in agent.optimizer.param_groups:
            group["lr"] = lr

        opponent = pick_opponent(ep / args.episodes)
        side = random.choice([1, -1])
        transitions, _ = play_episode(agent, opponent, side)

        for t in transitions:
            copies = WIN_COPIES if t[2] == 1.0 else 1
            m = mirror_transition(t)
            for _ in range(copies):
                agent.remember(*t)
                agent.remember(*m)
            if len(agent.memory) >= LEARN_START:
                loss = agent.learn()          # one learning step per real move
                if loss is not None:
                    losses.append(loss)

        agent.decay_epsilon()
        if ep % args.snapshot_every == 0:
            snapshot()
        if ep in save_at and ep != args.episodes:
            save_checkpoint(ep, f"ep_{ep}.pt", f"episode {ep}")
        if ep % args.eval_every == 0 or ep == args.episodes:
            run_eval(ep)

    # ---------- pick the final brain: best.pt or the last brain ----------
    print(f"\nFinal check: {FINAL_CHECK_GAMES} fresh games per opponent...")
    last_score = confirm_score(agent.policy_net, FINAL_CHECK_GAMES, seed=999)
    final_episode = args.episodes
    chosen = "last brain"
    best_path = os.path.join(out_dir, "best.pt")
    if best["episode"] is not None and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        best_net = DQN()
        best_net.load_state_dict(ckpt["policy"])
        best_net.eval()
        best_score = confirm_score(best_net, FINAL_CHECK_GAMES, seed=999)
        print(f"  last brain (episode {args.episodes}): score {last_score:.3f}")
        print(f"  best.pt   (episode {best['episode']}): score {best_score:.3f}")
        if best_score > last_score:
            agent.policy_net.load_state_dict(ckpt["policy"])
            agent.update_target()
            final_episode = best["episode"]
            chosen = f"best.pt from episode {best['episode']}"
    else:
        print(f"  last brain score {last_score:.3f}")
    print(f"  -> final.pt = {chosen}")

    save_checkpoint(final_episode, "final.pt", "final")
    metrics["final_choice"] = chosen
    metrics["finished"] = True
    write_json(metrics_path, metrics)
    print(f"\nDone in {(time.time() - start) / 60:.1f} minutes.")
    print("Saved:", ", ".join(c["file"] for c in metrics["checkpoints"]), "+ best.pt + metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=60)
    parser.add_argument("--snapshot-every", type=int, default=500)
    parser.add_argument("--save-at", default="0,500,5000,15000")
    parser.add_argument("--out-dir", default=os.path.join(HERE, "checkpoints"))
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())
