"""
Round-robin tournament between all the AI players, and the leaderboard.

Players: Trained AI (final.pt), Minimax, LLM (Groq), Random.
Every pair plays N games, alternating who moves first. Wins, losses and draws are added
up per player. Results are saved to ai/leaderboard.json (the backend and dashboard read it).
The Human row is added later by the web app from real games.

Run:  python ai/evaluate.py                       full leaderboard (100 games per pair, 6 per LLM pair)
      python ai/evaluate.py --games 40 --llm-games 4
      python ai/evaluate.py --no-llm              skip the LLM (no internet or no key)
      python ai/evaluate.py test                  self-tests
"""
import argparse
import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone

from connect4_env import Connect4
from opponents import AIPlayer, LLMPlayer, MinimaxPlayer, RandomPlayer, play_game

HERE = os.path.dirname(os.path.abspath(__file__))


def play_match(a, b, n_games, on_game=None):
    """Player `a` plays n_games against `b`, alternating who moves first."""
    a_wins = b_wins = draws = 0
    for i in range(n_games):
        if i % 2 == 0:
            result = play_game(a, b)        # a moves first: 1 means a won
        else:
            result = -play_game(b, a)       # b moves first: flip the sign so 1 still means a won
        if result == 1:
            a_wins += 1
        elif result == -1:
            b_wins += 1
        else:
            draws += 1
        if on_game:
            on_game(i + 1, n_games, a_wins, b_wins, draws)
    return {"a": a.name, "b": b.name, "a_wins": a_wins, "b_wins": b_wins,
            "draws": draws, "games": n_games}


def build_leaderboard(players, matches):
    """players = [(name, kind), ...]. Adds up every match into one row per player."""
    table = {name: {"name": name, "kind": kind, "wins": 0, "losses": 0, "draws": 0, "games": 0}
             for name, kind in players}
    for m in matches:
        a, b = table[m["a"]], table[m["b"]]
        a["wins"] += m["a_wins"]
        a["losses"] += m["b_wins"]
        b["wins"] += m["b_wins"]
        b["losses"] += m["a_wins"]
        for row in (a, b):
            row["draws"] += m["draws"]
            row["games"] += m["games"]
    rows = list(table.values())
    for row in rows:
        row["win_rate"] = round(row["wins"] / row["games"], 4) if row["games"] else 0.0
    rows.sort(key=lambda r: (-r["win_rate"], -r["wins"], r["name"]))
    return rows


class FallbackTracker:
    """Wraps the LLM player and counts WHY it had to fall back to a random move."""

    def __init__(self, llm):
        self.llm = llm
        self.name = llm.name
        self.reasons = {}

    def choose(self, game):
        column = self.llm.choose(game)
        if self.llm.last["fallback"]:
            reason = self.llm.last["reason"]
            self.reasons[reason] = self.reasons.get(reason, 0) + 1
        return column


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def evaluate(args):
    checkpoint = os.path.join(HERE, "checkpoints", args.checkpoint)
    ai = AIPlayer(checkpoint, name="Trained AI")
    minimax = MinimaxPlayer(depth=args.minimax_depth, seed=1)
    rnd = RandomPlayer(seed=2)
    players = [(ai, "ai"), (minimax, "minimax")]

    tracker = None
    if not args.no_llm:
        llm = LLMPlayer(seed=3)
        if llm.api_key:
            tracker = FallbackTracker(llm)
            players.append((tracker, "llm"))
        else:
            print("No GROQ_API_KEY found in .env, so the LLM is skipped.\n")
    players.append((rnd, "random"))

    print(f"Checkpoint: {args.checkpoint} (episode {ai.episode})")
    print(f"Players: {', '.join(p.name for p, _ in players)}\n")

    matches = []
    start = time.time()
    for (a, a_kind), (b, b_kind) in itertools.combinations(players, 2):
        is_llm = "llm" in (a_kind, b_kind)
        n = args.llm_games if is_llm else args.games
        print(f"{a.name} vs {b.name} ({n} games)")

        def progress(i, total, aw, bw, d):
            if is_llm:
                print(f"   game {i}/{total}: {aw}-{bw}-{d} (wins-losses-draws for {a.name})", flush=True)

        result = play_match(a, b, n, on_game=progress)
        matches.append(result)
        print(f"   -> {result['a_wins']} wins, {result['b_wins']} losses, {result['draws']} draws "
              f"for {a.name}  ({(time.time() - start) / 60:.1f} min)\n")

    rows = build_leaderboard([(p.name, k) for p, k in players], matches)

    print("LEADERBOARD")
    print(f"{'#':>2}  {'Player':<22}{'Wins':>6}{'Losses':>8}{'Draws':>7}{'Games':>7}{'Win rate':>10}")
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}  {r['name']:<22}{r['wins']:>6}{r['losses']:>8}{r['draws']:>7}"
              f"{r['games']:>7}{100 * r['win_rate']:>9.1f}%")

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": args.checkpoint,
        "checkpoint_episode": ai.episode,
        "games_per_pair": args.games,
        "llm_games_per_pair": args.llm_games if tracker else None,
        "players": rows,
        "matches": matches,
        "llm": None,
    }
    if tracker:
        calls, falls = tracker.llm.calls, tracker.llm.fallbacks
        data["llm"] = {
            "model": tracker.llm.model, "calls": calls, "fallbacks": falls,
            "fallback_rate": round(falls / calls, 4) if calls else 0.0,
            "fallback_reasons": tracker.reasons,
        }
        print(f"\nLLM: {calls} moves asked, {falls} fell back to a random move "
              f"({100 * falls / max(1, calls):.0f}%)")
        for reason, count in tracker.reasons.items():
            print(f"   {count} x {reason}")

    out = os.path.abspath(args.out)
    write_json(out, data)
    print(f"\nSaved: {out}")


# ---------- tests ----------

def run_tests():
    # 1. play_match: counts add up, and the win/loss sign is right whoever moves first
    result = play_match(MinimaxPlayer(depth=2, seed=1), RandomPlayer(seed=2), 10)
    assert result["a_wins"] + result["b_wins"] + result["draws"] == 10 == result["games"]
    assert result["a_wins"] >= 9, f"minimax should crush random, got {result}"
    result = play_match(RandomPlayer(seed=2), MinimaxPlayer(depth=2, seed=1), 10)
    assert result["b_wins"] >= 9, f"sign is wrong when the second player is stronger: {result}"
    print("PASS play_match counts and win/loss sign")

    # 2. leaderboard adds up wins, losses and draws for both sides of every match
    matches = [
        {"a": "A", "b": "B", "a_wins": 6, "b_wins": 3, "draws": 1, "games": 10},
        {"a": "A", "b": "C", "a_wins": 8, "b_wins": 2, "draws": 0, "games": 10},
        {"a": "B", "b": "C", "a_wins": 5, "b_wins": 4, "draws": 1, "games": 10},
    ]
    rows = build_leaderboard([("A", "ai"), ("B", "x"), ("C", "y")], matches)
    by = {r["name"]: r for r in rows}
    assert (by["A"]["wins"], by["A"]["losses"], by["A"]["draws"], by["A"]["games"]) == (14, 5, 1, 20)
    assert (by["B"]["wins"], by["B"]["losses"], by["B"]["draws"], by["B"]["games"]) == (8, 10, 2, 20)
    assert (by["C"]["wins"], by["C"]["losses"], by["C"]["draws"], by["C"]["games"]) == (6, 13, 1, 20)
    assert abs(by["A"]["win_rate"] - 0.7) < 1e-9 and rows[0]["name"] == "A" and rows[-1]["name"] == "C"
    assert sum(r["wins"] for r in rows) == sum(r["losses"] for r in rows), "wins and losses must match"
    print("PASS leaderboard totals and ordering")

    # 3. the fallback tracker counts why the LLM fell back
    class _Resp:
        status_code = 200
        headers = {}

        def json(self):
            return {"choices": [{"message": {"content": "no idea"}}]}

    llm = LLMPlayer(api_key="test", post=lambda *a, **k: _Resp(), seed=1)
    tracker = FallbackTracker(llm)
    game = Connect4()
    for _ in range(3):
        assert game.is_legal(tracker.choose(game))
    assert tracker.reasons == {"reply had no column number": 3}, tracker.reasons
    assert llm.calls == 3 and llm.fallbacks == 3
    print("PASS fallback tracker")

    # 4. the trained AI checkpoint loads
    path = os.path.join(HERE, "checkpoints", "final.pt")
    if os.path.exists(path):
        assert AIPlayer(path, name="Trained AI").name == "Trained AI"
        print("PASS trained AI checkpoint loads")
    else:
        print("SKIP checkpoint test (ai/checkpoints/final.pt not found)")

    print("\nAll tests passed.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--games", type=int, default=100, help="games per pair (no LLM)")
        parser.add_argument("--llm-games", type=int, default=6, help="games per pair that includes the LLM")
        parser.add_argument("--no-llm", action="store_true")
        parser.add_argument("--minimax-depth", type=int, default=2)
        parser.add_argument("--checkpoint", default="final.pt")
        parser.add_argument("--out", default=os.path.join(HERE, "leaderboard.json"))
        evaluate(parser.parse_args())
