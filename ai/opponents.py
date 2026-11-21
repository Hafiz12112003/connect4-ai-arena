"""
Opponents for Connect Four.

Every player has:
    name            - shown in the leaderboard
    choose(game)    - given a Connect4 game, returns the column to play (0-6)

Players here:
    RandomPlayer   - any legal column
    MinimaxPlayer  - alpha-beta search, configurable depth
    LLMPlayer      - asks an LLM on Groq (model set by GROQ_MODEL in .env). It paces itself to
                     stay under Groq's free-tier token limit. If a reply is still bad, it plays
                     a random legal move and FLAGS it (player.last["fallback"])
    AIPlayer       - your trained DQN, loaded from a checkpoint (.pt) file

Run this file directly to run the self-tests.
Run  python ai/opponents.py live  to ask the real LLM for one move (needs your key in .env).
"""
import json
import os
import random
import re
import sys
import time

import numpy as np
import requests
try:
    import torch
except ImportError:  # the deployed server runs without PyTorch
    torch = None
from dotenv import load_dotenv

from connect4_env import Connect4, ROWS, COLS
try:
    from model import DQN
except ImportError:  # the deployed server runs without PyTorch
    DQN = None

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, "..", ".env"))    # reads GROQ_API_KEY and GROQ_MODEL

WIN_SCORE = 100000
CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6]   # try middle columns first: better pruning
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"

LLM_MAX_TOKENS = 2000       # room for a reasoning model to think and then answer
LLM_MIN_BUDGET = 2300       # only ask when at least this many tokens are left this minute
LLM_MAX_ATTEMPTS = 5        # 1 try + up to 4 retries after a 429 (rate limit)
LLM_MAX_WAIT = 60.0         # never wait longer than this for one retry


def _build_windows():
    # Every possible line of 4 cells on the board, as flat indices (row * 7 + col).
    windows = []
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                end_r, end_c = r + 3 * dr, c + 3 * dc
                if 0 <= end_r < ROWS and 0 <= end_c < COLS:
                    windows.append([(r + i * dr) * COLS + (c + i * dc) for i in range(4)])
    return np.array(windows)


WINDOWS = _build_windows()   # 69 lines of four


def evaluate(board, me):
    """Score a board for player `me`. Positive = good for me, negative = good for them."""
    w = board.ravel()[WINDOWS]          # shape (69, 4): the discs in every line of four
    mine = (w == me).sum(axis=1)
    theirs = (w == -me).sum(axis=1)
    score = 0
    score += 5 * int(((mine == 3) & (theirs == 0)).sum())    # my three, one gap left
    score += 2 * int(((mine == 2) & (theirs == 0)).sum())    # my two, two gaps left
    score -= 4 * int(((theirs == 3) & (mine == 0)).sum())    # their three: dangerous
    score -= 1 * int(((theirs == 2) & (mine == 0)).sum())
    centre = board[:, 3]                                      # middle column is strong
    score += 3 * (int((centre == me).sum()) - int((centre == -me).sum()))
    return score


class RandomPlayer:
    def __init__(self, seed=None):
        self.name = "Random"
        self.rng = random.Random(seed)

    def choose(self, game):
        return self.rng.choice(game.legal_moves())


class MinimaxPlayer:
    """Looks `depth` moves ahead, assumes the opponent also plays well (minimax),
    and skips branches that cannot matter (alpha-beta pruning)."""

    def __init__(self, depth=3, seed=None):
        assert depth >= 1
        self.depth = depth
        self.name = f"Minimax (depth {depth})"
        self.rng = random.Random(seed)

    def _ordered_moves(self, game):
        legal = set(game.legal_moves())
        return [c for c in CENTER_ORDER if c in legal]

    def _search(self, game, depth, alpha, beta, me):
        if game.done:
            if game.winner == 0:
                return 0
            bonus = WIN_SCORE + depth            # a faster win scores higher
            return bonus if game.winner == me else -bonus
        if depth == 0:
            return evaluate(game.board, me)

        if game.current_player == me:            # my turn: pick the highest score
            value = -float("inf")
            for col in self._ordered_moves(game):
                child = game.copy()
                child.step(col)
                value = max(value, self._search(child, depth - 1, alpha, beta, me))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break                        # opponent would never allow this branch
            return value
        else:                                    # their turn: they pick the lowest score
            value = float("inf")
            for col in self._ordered_moves(game):
                child = game.copy()
                child.step(col)
                value = min(value, self._search(child, depth - 1, alpha, beta, me))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def choose(self, game):
        me = game.current_player
        best = -float("inf")
        best_moves = []
        for col in self._ordered_moves(game):
            child = game.copy()
            child.step(col)
            # alpha = best - 1 means: only moves at least as good as the best so far
            # get an exact score, so equally good moves can be picked at random.
            score = self._search(child, self.depth - 1, best - 1, float("inf"), me)
            if score > best:
                best, best_moves = score, [col]
            elif score == best:
                best_moves.append(col)
        return self.rng.choice(best_moves)


# ---------- the trained DQN as a player ----------

class AIPlayer:
    """Your trained DQN, loaded from a checkpoint file. Always plays the best legal column."""

    def __init__(self, checkpoint, name="Trained AI"):
        self.name = name
        self.net = DQN()
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.net.load_state_dict(ckpt["policy"])
        self.net.eval()
        self.episode = ckpt.get("extra", {}).get("episode")

    def q_values(self, game):
        with torch.no_grad():
            x = torch.as_tensor(game.get_state()).unsqueeze(0)
            return self.net(x).squeeze(0).numpy()

    def choose(self, game):
        q = np.where(game.legal_mask() > 0, self.q_values(game), -np.inf)   # never a full column
        return int(np.argmax(q))


# ---------- the LLM as a player ----------

SYSTEM_PROMPT = (
    "You are playing Connect Four. Discs fall to the lowest empty cell of a column. "
    "You win by getting four of your discs in a row: horizontal, vertical or diagonal. "
    "Reply with ONLY a JSON object like {\"column\": 3}. No other text."
)


def parse_column(text):
    """Pull the column number out of the model's reply. Returns an int, or None."""
    if not text:
        return None
    m = re.search(r'"column"\s*:\s*(-?\d+)', text)
    if m:
        return int(m.group(1))
    m = re.search(r'-?\d+', text)
    return int(m.group(0)) if m else None


def parse_duration(text):
    """Turn Groq's time strings ('19.822s', '1m2.5s', '250ms', '3') into seconds. None if unreadable."""
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    parts = re.findall(r'(\d+(?:\.\d+)?)(ms|h|m|s)', text)
    if not parts:
        return None
    unit = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
    return sum(float(number) * unit[u] for number, u in parts)


def board_prompt(game):
    """The board as text, from the point of view of the player to move (X = me, O = them)."""
    me = game.current_player
    symbols = {0: ".", me: "X", -me: "O"}
    rows = [" ".join(symbols[int(v)] for v in row) for row in game.board]
    rows.append("0 1 2 3 4 5 6")
    return (
        "Board (top row first). You are X, your opponent is O, '.' is empty:\n"
        + "\n".join(rows)
        + f"\n\nColumns you may play: {game.legal_moves()}\n"
        'Reply with ONLY JSON, for example {"column": 3}.'
    )


def _header(resp, name):
    """Read one response header, ignoring upper/lower case. None if it is missing."""
    try:
        for key, value in resp.headers.items():
            if key.lower() == name:
                return value
    except Exception:
        pass
    return None


class LLMPlayer:
    """Asks an LLM on Groq for its move. It paces itself to stay under the free-tier token limit,
    and waits out rate-limit errors (429). If the request still fails, or the reply is not a
    legal column, it plays a random legal move and sets last["fallback"] = True."""

    def __init__(self, api_key=None, model=None, post=None, timeout=30, seed=None, sleep=None):
        key = api_key if api_key is not None else os.getenv("GROQ_API_KEY", "")
        self.api_key = "" if key in ("", "paste_your_key_here") else key
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self.name = f"LLM ({self.model.split('/')[-1]})"
        self.timeout = timeout
        self._post = post or requests.post       # can be swapped for a fake in tests
        self._sleep = sleep or time.sleep        # can be swapped for a fake in tests
        self.rng = random.Random(seed)
        self.calls = 0
        self.fallbacks = 0
        self.last = {"fallback": False, "reason": "", "raw": "", "column": None}
        # what Groq last told us about our token budget for this minute
        self._tokens_left = None
        self._refill_seconds = None
        self._budget_time = None

    def _update_budget(self, resp):
        left = _header(resp, "x-ratelimit-remaining-tokens")
        refill = parse_duration(_header(resp, "x-ratelimit-reset-tokens"))
        try:
            self._tokens_left = int(float(left)) if left is not None else None
        except ValueError:
            self._tokens_left = None
        self._refill_seconds = refill
        self._budget_time = time.time()

    def _wait_for_budget(self):
        """If the token budget for this minute is too low for another move, wait for the refill."""
        if self._tokens_left is None or self._refill_seconds is None:
            return
        if self._tokens_left >= LLM_MIN_BUDGET:
            return
        elapsed = time.time() - self._budget_time
        wait = min(self._refill_seconds - elapsed + 0.5, LLM_MAX_WAIT)
        if wait > 0:
            self._sleep(wait)
        self._tokens_left = None       # the refill has happened: stop waiting until Groq says otherwise

    def _retry_wait(self, resp):
        wait = parse_duration(_header(resp, "retry-after"))
        if wait is None:
            wait = parse_duration(_header(resp, "x-ratelimit-reset-tokens"))
        if wait is None:
            wait = 5.0
        return min(max(wait, 0.0) + 0.25, LLM_MAX_WAIT)

    def _ask(self, game, legal):
        """Returns (column or None, reason it failed, raw reply text)."""
        if not self.api_key:
            return None, "no API key set", ""
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": board_prompt(game)},
            ],
            "temperature": 0,
            "max_tokens": LLM_MAX_TOKENS,   # reasoning models think a lot first (about 900 tokens per move)
        }
        if self.model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = "low"     # less thinking = faster and cheaper
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        self._wait_for_budget()
        resp = None
        for attempt in range(LLM_MAX_ATTEMPTS):
            try:
                resp = self._post(GROQ_URL, headers=headers, json=body, timeout=self.timeout)
            except Exception as e:
                return None, f"request failed ({type(e).__name__})", ""
            self._update_budget(resp)
            if resp.status_code == 429 and attempt < LLM_MAX_ATTEMPTS - 1:
                self._sleep(self._retry_wait(resp))      # rate limited: wait as long as Groq asks, then retry
                continue
            break
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}", ""
        try:
            raw = resp.json()["choices"][0]["message"]["content"] or ""
        except Exception:
            return None, "unexpected response shape", ""
        col = parse_column(raw)
        if col is None:
            return None, "reply had no column number", raw
        if col not in legal:
            return None, f"column {col} is not a legal move", raw
        return col, "", raw

    def choose(self, game):
        legal = game.legal_moves()
        self.calls += 1
        column, reason, raw = self._ask(game, legal)
        if column is None:
            self.fallbacks += 1
            column = self.rng.choice(legal)
            self.last = {"fallback": True, "reason": reason, "raw": raw, "column": column}
        else:
            self.last = {"fallback": False, "reason": "", "raw": raw, "column": column}
        return column


def play_game(first, second):
    """Play one full game. Returns 1 if `first` won, -1 if `second` won, 0 for a draw."""
    game = Connect4()
    players = {1: first, -1: second}
    while not game.done:
        game.step(players[game.current_player].choose(game))
    return game.winner


# ---------- tests ----------

class _FakeResponse:
    def __init__(self, status=200, content=None, headers=None):
        self.status_code = status
        self.headers = headers or {}
        self._content = content

    def json(self):
        if self._content is None:
            raise ValueError("no json")
        return {"choices": [{"message": {"content": self._content}}]}


def _fake_post(items):
    """A fake requests.post that returns (or raises) the given items one by one."""
    queue = list(items)
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    post.calls = calls
    return post


def run_tests():
    def setup(moves):
        g = Connect4()
        for m in moves:
            g.step(m)
        return g

    # 1. Window table: a 6x7 board has exactly 69 lines of four
    assert len(WINDOWS) == 69, f"expected 69 windows, got {len(WINDOWS)}"
    print("PASS 69 lines of four")

    # 2. Random player only plays legal columns
    g = setup([0] * 6)                      # column 0 is full
    rp = RandomPlayer(seed=1)
    assert all(rp.choose(g) in (1, 2, 3, 4, 5, 6) for _ in range(200)), "random played a full column"
    print("PASS random player is legal")

    # 3. Minimax takes an immediate win (X has 0,1,2 on the bottom row, X to move)
    g = setup([0, 6, 1, 6, 2, 6])
    assert MinimaxPlayer(depth=2, seed=1).choose(g) == 3, "minimax missed a win"
    print("PASS minimax takes a win")

    # 4. Minimax blocks (O has 0,1,2 on the bottom row, X to move and must block column 3)
    g = setup([6, 0, 6, 1, 5, 2])
    assert MinimaxPlayer(depth=2, seed=1).choose(g) == 3, "minimax missed a block"
    print("PASS minimax blocks a loss")

    # 5. Minimax (depth 3) should crush the random player, going first and second
    mm = MinimaxPlayer(depth=3, seed=1)
    rp = RandomPlayer(seed=2)
    wins = 0
    for i in range(20):
        if i % 2 == 0:
            wins += play_game(mm, rp) == 1
        else:
            wins += play_game(rp, mm) == -1
    assert wins >= 18, f"minimax only won {wins}/20 against random"
    print(f"PASS minimax beat random {wins}/20 (this test takes a few seconds)")

    # 6. Reading the column out of the LLM's reply
    assert parse_column('{"column": 3}') == 3
    assert parse_column('```json\n{"column": 5}\n```') == 5
    assert parse_column("I will play column 4.") == 4
    assert parse_column("no idea") is None and parse_column("") is None
    print("PASS parsing the LLM reply")

    # 7. The prompt shows the board from the mover's point of view
    g = setup([3])                          # player 1 played column 3, so player -1 is to move
    text = board_prompt(g)
    assert "O" in text and "X" not in text.split("Columns")[0].replace("X, your", ""), "perspective wrong"
    assert "[0, 1, 2, 3, 4, 5, 6]" in text
    print("PASS prompt perspective")

    # 8. LLM player with a fake server: good reply, illegal reply, garbage, errors
    g = setup([0] * 6)                      # column 0 is full
    no_sleep = lambda seconds: None

    p = LLMPlayer(api_key="test", post=_fake_post([_FakeResponse(content='{"column": 3}')]), sleep=no_sleep, seed=1)
    assert p.choose(g) == 3 and not p.last["fallback"], "valid reply should be used"

    p = LLMPlayer(api_key="test", post=_fake_post([_FakeResponse(content='{"column": 0}')]), sleep=no_sleep, seed=1)
    move = p.choose(g)
    assert p.last["fallback"] and "not a legal" in p.last["reason"] and g.is_legal(move), "illegal reply must fall back"

    p = LLMPlayer(api_key="test", post=_fake_post([_FakeResponse(content="hmm, tough one")]), sleep=no_sleep, seed=1)
    assert g.is_legal(p.choose(g)) and p.last["fallback"], "garbage reply must fall back"

    p = LLMPlayer(api_key="test", post=_fake_post([_FakeResponse(content="")]), sleep=no_sleep, seed=1)
    assert g.is_legal(p.choose(g)) and p.last["fallback"], "empty reply must fall back"

    p = LLMPlayer(api_key="test", post=_fake_post([_FakeResponse(status=500)]), sleep=no_sleep, seed=1)
    assert g.is_legal(p.choose(g)) and p.last["reason"] == "HTTP 500", "server error must fall back"

    p = LLMPlayer(api_key="test", post=_fake_post([requests.exceptions.Timeout()]), sleep=no_sleep, seed=1)
    assert g.is_legal(p.choose(g)) and p.last["fallback"], "timeout must fall back"

    never = _fake_post([])
    p = LLMPlayer(api_key="", post=never, sleep=no_sleep, seed=1)
    assert g.is_legal(p.choose(g)) and p.last["reason"] == "no API key set" and not never.calls, "no key must not call"
    print("PASS LLM player: valid use, and flagged random fallback on every kind of failure")

    # 9. Reading Groq's time strings
    assert parse_duration("19.822s") == 19.822 and parse_duration("250ms") == 0.25
    assert parse_duration("1m2.5s") == 62.5 and parse_duration("2h9m36s") == 7776.0
    assert parse_duration("3") == 3.0 and parse_duration("") is None and parse_duration(None) is None
    assert parse_duration("soon") is None
    print("PASS reading Groq time strings")

    # 10. Rate limits: wait as long as Groq says, retry, and give up (flagged) only after 5 tries
    sleeps = []
    p = LLMPlayer(api_key="test", post=_fake_post([
        _FakeResponse(status=429, headers={"retry-after": "3"}),
        _FakeResponse(content='{"column": 2}')]), sleep=sleeps.append, seed=1)
    assert p.choose(g) == 2 and not p.last["fallback"], "a 429 should be retried"
    assert len(sleeps) == 1 and 3.0 <= sleeps[0] <= 3.5, f"should wait about 3 seconds, slept {sleeps}"
    assert p.calls == 1 and p.fallbacks == 0

    sleeps = []
    p = LLMPlayer(api_key="test", post=_fake_post(
        [_FakeResponse(status=429, headers={"retry-after": "1"}) for _ in range(5)]), sleep=sleeps.append, seed=1)
    assert g.is_legal(p.choose(g)) and p.last["reason"] == "HTTP 429", "5 rate limits in a row must fall back"
    assert len(sleeps) == 4, f"should wait between tries, slept {sleeps}"
    print("PASS rate limits: waits as Groq asks, retries, flags only after 5 tries")

    # 11. Pacing: when Groq says the token budget is nearly used up, wait for the refill first
    sleeps = []
    low_budget = {"x-ratelimit-remaining-tokens": "100", "x-ratelimit-reset-tokens": "10s"}
    fake = _fake_post([_FakeResponse(content='{"column": 3}', headers=low_budget),
                       _FakeResponse(content='{"column": 4}')])
    p = LLMPlayer(api_key="test", post=fake, sleep=sleeps.append, seed=1)
    fresh = Connect4()
    p.choose(fresh)
    assert sleeps == [], "nothing is known on the first call, so no waiting"
    p.choose(fresh)
    assert len(sleeps) == 1 and 9.0 <= sleeps[0] <= 11.0, f"should wait for the refill, slept {sleeps}"

    sleeps = []
    plenty = {"x-ratelimit-remaining-tokens": "7000", "x-ratelimit-reset-tokens": "10s"}
    fake = _fake_post([_FakeResponse(content='{"column": 3}', headers=plenty),
                       _FakeResponse(content='{"column": 4}')])
    p = LLMPlayer(api_key="test", post=fake, sleep=sleeps.append, seed=1)
    p.choose(fresh)
    p.choose(fresh)
    assert sleeps == [], "plenty of budget left: no waiting"
    print("PASS pacing: waits for the token refill only when the budget is low")

    # 12. The request itself: enough tokens for a reasoning model, and settings only where they fit
    fake = _fake_post([_FakeResponse(content='{"column": 3}')])
    p = LLMPlayer(api_key="test", model="openai/gpt-oss-20b", post=fake, sleep=no_sleep, seed=1)
    p.choose(g)
    sent = fake.calls[0]["json"]
    assert sent["max_tokens"] == LLM_MAX_TOKENS >= 1500 and sent["reasoning_effort"] == "low"
    assert sent["model"] == "openai/gpt-oss-20b"
    assert p.name == "LLM (gpt-oss-20b)", f"unexpected name {p.name}"
    fake = _fake_post([_FakeResponse(content='{"column": 3}')])
    p = LLMPlayer(api_key="test", model="qwen/qwen3.8-27b", post=fake, sleep=no_sleep, seed=1)
    p.choose(g)
    assert "reasoning_effort" not in fake.calls[0]["json"], "reasoning_effort should only go to gpt-oss models"
    print("PASS request settings (token room, reasoning effort only for gpt-oss)")

    # 13. The trained AI loads from a checkpoint, plays legal moves and beats Random
    path = os.path.join(HERE, "checkpoints", "final.pt")
    if os.path.exists(path):
        ai = AIPlayer(path)
        g = setup([0] * 6)
        assert ai.choose(g) != 0, "AI picked a full column"
        rp = RandomPlayer(seed=5)
        wins = 0
        for i in range(20):
            wins += (play_game(ai, rp) == 1) if i % 2 == 0 else (play_game(rp, ai) == -1)
        assert wins >= 16, f"trained AI only won {wins}/20 against random"
        print(f"PASS trained AI loads from final.pt and beat random {wins}/20")
    else:
        print("SKIP trained AI test (ai/checkpoints/final.pt not found)")

    print("\nAll tests passed.")


def live_check():
    """Ask the real LLM for one move, to check your key and model work."""
    p = LLMPlayer()
    print(f"Model: {p.model}")
    print(f"API key set: {bool(p.api_key)}")
    g = Connect4()
    for m in [3, 3, 2]:
        g.step(m)
    print("\nBoard:")
    print(g.render())
    move = p.choose(g)
    print(f"\nLLM reply : {p.last['raw']!r}")
    print(f"LLM column: {move}")
    if p.last["fallback"]:
        print(f"FALLBACK to a random move. Reason: {p.last['reason']}")
    else:
        print("The LLM answered with a legal move. Your key works.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "live":
        live_check()
    else:
        run_tests()
