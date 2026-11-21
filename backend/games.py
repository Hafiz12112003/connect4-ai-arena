import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AI_DIR = os.path.join(ROOT, "ai")
for p in (HERE, AI_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from brain_np import get_brain  # noqa: E402
from connect4_env import Connect4  # noqa: E402
from opponents import MinimaxPlayer  # noqa: E402

# Final difficulty mapping (tested in tune_variety.py and tune_opening.py).
LEVELS = {
    "easy": "ep_500.pt",
    "medium": "ep_5000.pt",
    "hard": "final.pt",
}


def board_state(game):
    """Everything the browser needs to draw the board."""
    last = game.last_move
    return {
        "board": [[int(v) for v in row] for row in game.board],
        "current_player": int(game.current_player),
        "winner": int(game.winner),
        "done": bool(game.done),
        "draw": bool(game.done and game.winner == 0),
        "winning_cells": [[int(r), int(c)] for r, c in game.winning_cells],
        "legal": [int(c) for c in game.legal_moves()],
        "last_move": [int(last[0]), int(last[1])] if last else None,
        "move_count": int(game.move_count),
    }


def make_event(game, who, player, column, brain=None, llm=None):
    """One move, described for the browser."""
    row = int(game.last_move[0])
    return {
        "who": who,            # "human", "ai", "minimax" or "llm"
        "player": int(player), # 1 or -1
        "column": int(column),
        "row": row,
        "brain": brain,        # real Q-values and activations (AI moves only)
        "llm": llm,            # fallback info (LLM moves only)
        "state": board_state(game),
    }


class Session:
    """Human vs AI. The human picks the colour order, the AI is greedy."""

    def __init__(self, difficulty="medium", human_first=True):
        if difficulty not in LEVELS:
            raise ValueError(f"unknown difficulty: {difficulty}")
        self.difficulty = difficulty
        self.game = Connect4()
        self.human = 1 if human_first else -1
        self.ai = -self.human
        self.brain = get_brain(LEVELS[difficulty])

    def state(self):
        s = board_state(self.game)
        s.update({"mode": "human_ai", "difficulty": self.difficulty,
                  "human": self.human, "ai": self.ai})
        return s

    def ai_to_move(self):
        return (not self.game.done) and self.game.current_player == self.ai

    def human_move(self, col):
        if self.game.done:
            raise ValueError("the game is over")
        if self.game.current_player != self.human:
            raise ValueError("not your turn")
        try:
            col = int(col)
        except (TypeError, ValueError):
            raise ValueError("column must be a number")
        if not 0 <= col <= 6 or not self.game.is_legal(col):
            raise ValueError("illegal column")
        self.game.step(col)
        return make_event(self.game, "human", self.human, col)

    def ai_move(self):
        if not self.ai_to_move():
            raise ValueError("it is not the AI's turn")
        info = self.brain.think(self.game)
        col = info["column"]
        self.game.step(col)
        return make_event(self.game, "ai", self.ai, col, brain=info)


class Match:
    """AI vs Minimax or AI vs LLM. Call next() to get one move at a time."""

    def __init__(self, opponent="minimax2", ai_first=True,
                 difficulty="hard", seed=None):
        if difficulty not in LEVELS:
            raise ValueError(f"unknown difficulty: {difficulty}")
        self.opponent = opponent
        self.difficulty = difficulty
        self.game = Connect4()
        self.ai = 1 if ai_first else -1
        self.brain = get_brain(LEVELS[difficulty])
        if seed is None:
            seed = random.randrange(1_000_000)
        if opponent.startswith("minimax"):
            self.who = "minimax"
            self.player = MinimaxPlayer(int(opponent[len("minimax"):]), seed=seed)
        elif opponent == "llm":
            self.who = "llm"
            from opponents import LLMPlayer  # loaded only when needed
            self.player = LLMPlayer()
        else:
            raise ValueError(f"unknown opponent: {opponent}")

    def state(self):
        s = board_state(self.game)
        s.update({"mode": f"ai_vs_{self.opponent}", "difficulty": self.difficulty,
                  "ai": self.ai, "opponent": self.opponent})
        return s

    def next_is_llm(self):
        return (not self.game.done and self.who == "llm"
                and self.game.current_player != self.ai)

    def next(self):
        if self.game.done:
            raise ValueError("the game is over")
        mover = self.game.current_player
        if mover == self.ai:
            info = self.brain.think(self.game)
            self.game.step(info["column"])
            return make_event(self.game, "ai", mover, info["column"], brain=info)
        col = self.player.choose(self.game)
        llm = None
        if self.who == "llm":
            last = getattr(self.player, "last", None)
            llm = dict(last) if last else None
        self.game.step(col)
        return make_event(self.game, self.who, mover, col, llm=llm)


def _self_test():
    print("== Levels ==")
    for level, fname in LEVELS.items():
        assert os.path.exists(os.path.join(AI_DIR, "checkpoints", fname)), fname
        print(f"{level:7s} -> {fname}")

    print()
    print("== Rule checks ==")
    s = Session("medium", human_first=True)
    for bad in (7, -1, "x"):
        try:
            s.human_move(bad)
            raise AssertionError(f"bad column {bad!r} was accepted")
        except ValueError:
            pass
    try:
        s.ai_move()
        raise AssertionError("AI moved on the human's turn")
    except ValueError:
        pass
    s.human_move(3)
    try:
        s.human_move(3)
        raise AssertionError("human moved twice in a row")
    except ValueError:
        pass
    ev = s.ai_move()
    assert ev["who"] == "ai" and len(ev["brain"]["q_values"]) == 7
    print("bad columns, wrong turn, double move: all refused. OK")

    print()
    print("== Scripted human vs AI (hard) ==")
    prefer = [3, 2, 4, 1, 5, 0, 6]
    for human_first in (True, False):
        s = Session("hard", human_first=human_first)
        n = 0
        while not s.game.done:
            if s.ai_to_move():
                s.ai_move()
            else:
                col = next(c for c in prefer if s.game.is_legal(c))
                s.human_move(col)
            n += 1
        st = s.state()
        if st["winner"] == 0:
            result = "draw"
        else:
            result = "human wins" if st["winner"] == s.human else "AI wins"
        print(f"human_first={human_first!s:5s}: {result} after {n} moves, "
              f"winning cells: {len(st['winning_cells'])}")

    print()
    print("== AI (hard) vs Minimax depth 2, both colours ==")
    for ai_first in (True, False):
        m = Match("minimax2", ai_first=ai_first, difficulty="hard", seed=0)
        ai_events = 0
        while not m.game.done:
            ev = m.next()
            if ev["who"] == "ai":
                b = ev["brain"]
                assert len(b["q_values"]) == 7 and len(b["input"]) == 84
                assert len(b["hidden1"]) == 12 and len(b["hidden2"]) == 12
                ai_events += 1
            else:
                assert ev["who"] == "minimax" and ev["brain"] is None
        w = m.game.winner
        result = "draw" if w == 0 else ("AI wins" if w == m.ai else "Minimax wins")
        print(f"ai_first={ai_first!s:5s}: {result} | AI moves with full brain data: {ai_events}")

    print("Self-test finished.")


if __name__ == "__main__":
    _self_test()
