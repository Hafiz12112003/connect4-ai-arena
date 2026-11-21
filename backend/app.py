import itertools
import json
import os
import sys
import threading
import time

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AI_DIR = os.path.join(ROOT, "ai")
for p in (HERE, AI_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
load_dotenv(os.path.join(ROOT, ".env"))

from brain_np import get_brain  # noqa: E402
from games import LEVELS, Match, Session, board_state  # noqa: E402

LEADERBOARD_PATH = os.path.join(AI_DIR, "leaderboard.json")
METRICS_PATH = os.path.join(AI_DIR, "checkpoints", "metrics.json")
HUMAN_STATS_PATH = os.path.join(HERE, "human_stats.json")

THINK_DELAY = 0.6   # seconds the "thinking" pulse lasts before an AI move
MOVE_PAUSE = 1.0    # seconds between moves in spectator matches (divided by speed)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


# The LLM runs on a free API with a small daily quota, so a public server limits it.
LLM_ENABLED = os.environ.get("LLM_ENABLED", "1") != "0"
LLM_DAILY_CAP = _env_int("LLM_DAILY_CAP", 200)
LLM_OFF_MESSAGE = "AI vs LLM is switched off on this server. Try AI vs Minimax instead."
LLM_LIMIT_MESSAGE = (
    "The LLM demo has reached its daily limit (it uses a free API with a small quota). "
    "Try AI vs Minimax, or come back tomorrow."
)

origins = os.environ.get("CORS_ORIGINS", "*")
origins = "*" if origins == "*" else [o.strip() for o in origins.split(",")]

app = Flask(__name__)
CORS(app, origins=origins)
socketio = SocketIO(app, cors_allowed_origins=origins, async_mode="threading")

# Load every brain once at start-up. The deployed server never runs the __main__ block.
for _name in LEVELS.values():
    get_brain(_name)

sessions = {}      # sid -> Session (Human vs AI)
matches = {}       # sid -> {"id", "match", "speed"}
match_counts = {}  # sid -> how many matches started (used to alternate colours)
_ids = itertools.count(1)
_stats_lock = threading.Lock()
_llm_lock = threading.Lock()
_llm_day = {"day": None, "used": 0}


# ---------- small helpers ----------

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_human_stats():
    data = load_json(HUMAN_STATS_PATH) or {}
    return {k: int(data.get(k, 0)) for k in ("wins", "losses", "draws")}


def record_human_result(result):
    key = {"win": "wins", "loss": "losses", "draw": "draws"}[result]
    with _stats_lock:
        stats = read_human_stats()
        stats[key] += 1
        with open(HUMAN_STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(stats, f)


def _llm_roll_day():
    day = time.strftime("%Y-%m-%d", time.gmtime())
    if _llm_day["day"] != day:
        _llm_day["day"] = day
        _llm_day["used"] = 0


def llm_remaining():
    """LLM moves still allowed today (the day rolls over at midnight UTC)."""
    with _llm_lock:
        _llm_roll_day()
        return max(LLM_DAILY_CAP - _llm_day["used"], 0)


def llm_try_spend():
    """Use up one LLM move from today's allowance. False when none is left."""
    with _llm_lock:
        _llm_roll_day()
        if _llm_day["used"] >= LLM_DAILY_CAP:
            return False
        _llm_day["used"] += 1
        return True


def clamp_speed(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 1.0
    return min(max(v, 0.25), 20.0)


def game_over_payload(game, winner_who):
    st = board_state(game)
    return {"winner": st["winner"], "draw": st["draw"],
            "winning_cells": st["winning_cells"],
            "winner_who": winner_who, "state": st}


def send_error(sid, message):
    socketio.emit("error", {"message": message}, to=sid)


# ---------- REST ----------

@app.get("/api/health")
def health():
    return jsonify({"ok": True})


@app.get("/api/difficulties")
def difficulties():
    notes = {
        "easy": "Early brain (500 games of training). Beats Random about 87%, loses to Minimax depth 1.",
        "medium": "Mid-training brain. Beats Minimax depth 1 about 58%, loses to Minimax depth 2.",
        "hard": "Final brain. Beats Minimax depth 1 about 77%. Plays deterministically.",
    }
    return jsonify([{"id": k, "label": k.capitalize(), "checkpoint": v, "note": notes[k]}
                    for k, v in LEVELS.items()])


@app.get("/api/leaderboard")
def leaderboard():
    data = load_json(LEADERBOARD_PATH)
    if data is None:
        return jsonify({"error": "leaderboard.json not found. Run ai/evaluate.py first."}), 404
    players = list(data.get("players", []))
    # Match the scale used in the saved file (fraction or percent).
    scale = 100 if any((p.get("win_rate") or 0) > 1 for p in players) else 1
    h = read_human_stats()
    games = sum(h.values())
    players.append({
        "name": "Human", "kind": "human",
        "wins": h["wins"], "losses": h["losses"], "draws": h["draws"],
        "games": games,
        "win_rate": (h["wins"] / games * scale) if games else None,
    })
    data = dict(data)
    data["players"] = players
    return jsonify(data)


@app.get("/api/metrics")
def metrics():
    data = load_json(METRICS_PATH)
    if data is None:
        return jsonify({"error": "metrics.json not found. Run ai/train.py first."}), 404
    return jsonify(data)


# ---------- Human vs AI ----------

def ai_turn(sid, s):
    socketio.emit("ai_thinking", {"who": "ai", "player": s.ai}, to=sid)
    socketio.sleep(THINK_DELAY)
    ev = s.ai_move()
    socketio.emit("ai_move", ev, to=sid)
    if s.game.done:
        finish_session(sid, s)


def finish_session(sid, s):
    g = s.game
    if g.winner == 0:
        result, who = "draw", None
    elif g.winner == s.human:
        result, who = "win", "human"
    else:
        result, who = "loss", "ai"
    record_human_result(result)
    socketio.emit("game_over", game_over_payload(g, who), to=sid)
    sessions.pop(sid, None)


@socketio.on("new_game")
def on_new_game(data=None):
    data = data or {}
    sid = request.sid
    matches.pop(sid, None)
    try:
        s = Session(data.get("difficulty", "medium"), bool(data.get("human_first", True)))
    except ValueError as e:
        send_error(sid, str(e))
        return
    sessions[sid] = s
    socketio.emit("game_state", s.state(), to=sid)
    if s.ai_to_move():
        ai_turn(sid, s)


@socketio.on("human_move")
def on_human_move(data=None):
    sid = request.sid
    s = sessions.get(sid)
    if s is None:
        send_error(sid, "no game is running")
        return
    try:
        s.human_move((data or {}).get("col"))
    except ValueError as e:
        send_error(sid, str(e))
        return
    socketio.emit("game_state", s.state(), to=sid)
    if s.game.done:
        finish_session(sid, s)
    else:
        ai_turn(sid, s)


# ---------- Spectator matches ----------

def run_match(sid, match_id):
    def current():
        m = matches.get(sid)
        return m if (m and m["id"] == match_id) else None

    try:
        while True:
            m = current()
            if m is None:
                return
            match = m["match"]
            if match.game.done:
                break
            speed = m["speed"]
            if match.game.current_player == match.ai:
                socketio.emit("ai_thinking", {"who": "ai", "player": match.ai}, to=sid)
                socketio.sleep(THINK_DELAY / speed)
            elif match.next_is_llm():
                if not llm_try_spend():
                    # Out of LLM moves for today: end the match cleanly, no LLM call is made.
                    send_error(sid, LLM_LIMIT_MESSAGE)
                    socketio.emit("match_stopped", {}, to=sid)
                    if current() is not None:
                        matches.pop(sid, None)
                    return
                socketio.emit("ai_thinking", {"who": "llm", "player": -match.ai,
                                              "message": "LLM is thinking..."}, to=sid)
            if current() is None:
                return
            ev = match.next()  # an LLM move can take 8-9 seconds
            if current() is None:
                return
            socketio.emit("ai_move", ev, to=sid)
            if match.game.done:
                break
            socketio.sleep(MOVE_PAUSE / m["speed"])
        g = match.game
        if g.winner == 0:
            who = None
        elif g.winner == match.ai:
            who = "ai"
        else:
            who = match.who
        socketio.emit("game_over", game_over_payload(g, who), to=sid)
        if current() is not None:
            matches.pop(sid, None)
    except Exception as e:  # noqa: BLE001
        send_error(sid, f"match failed: {e}")
        if current() is not None:
            matches.pop(sid, None)


@socketio.on("start_match")
def on_start_match(data=None):
    data = data or {}
    sid = request.sid
    matches.pop(sid, None)
    sessions.pop(sid, None)
    opponent = str(data.get("opponent", "minimax2"))
    if opponent == "llm":
        if not LLM_ENABLED:
            send_error(sid, LLM_OFF_MESSAGE)
            return
        if llm_remaining() <= 0:
            send_error(sid, LLM_LIMIT_MESSAGE)
            return
    count = match_counts.get(sid, 0)
    ai_first = data.get("ai_first")
    if ai_first is None:
        ai_first = (count % 2 == 0)  # alternate colours match after match
    try:
        m = Match(opponent, bool(ai_first), data.get("difficulty", "hard"))
    except Exception as e:  # noqa: BLE001
        send_error(sid, f"could not start match: {e}")
        return
    match_counts[sid] = count + 1
    match_id = next(_ids)
    matches[sid] = {"id": match_id, "match": m, "speed": clamp_speed(data.get("speed", 1))}
    socketio.emit("game_state", m.state(), to=sid)
    socketio.start_background_task(run_match, sid, match_id)


@socketio.on("set_speed")
def on_set_speed(data=None):
    m = matches.get(request.sid)
    if m:
        m["speed"] = clamp_speed((data or {}).get("speed", 1))


@socketio.on("stop_match")
def on_stop_match(data=None):
    matches.pop(request.sid, None)
    socketio.emit("match_stopped", {}, to=request.sid)


@socketio.on("disconnect")
def on_disconnect(*args):
    sid = request.sid
    matches.pop(sid, None)
    sessions.pop(sid, None)
    match_counts.pop(sid, None)


@socketio.on_error_default
def on_error(e):
    send_error(request.sid, str(e))


# ---------- self-tests ----------

def _start_test_server(port):
    import logging
    import requests
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    t = threading.Thread(
        target=lambda: socketio.run(app, host="127.0.0.1", port=port,
                                    allow_unsafe_werkzeug=True),
        daemon=True)
    t.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            requests.get(url + "/api/health", timeout=1)
            return url
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    raise RuntimeError("test server did not start")


class _TestClient:
    def __init__(self, url):
        import queue
        import socketio as sio_client
        self.queue = queue
        self.q = queue.Queue()
        self.sio = sio_client.Client()
        self.sio.on("*", self._on_any)
        self.sio.connect(url, transports=["polling"])

    def _on_any(self, event, data=None):
        self.q.put((event, data))

    def send(self, event, data=None):
        self.sio.emit(event, data)

    def drain(self):
        while True:
            try:
                self.q.get_nowait()
            except self.queue.Empty:
                return

    def wait(self, names, timeout=20, required=True):
        deadline = time.time() + timeout
        while True:
            left = deadline - time.time()
            if left <= 0:
                if required:
                    raise AssertionError(f"timed out waiting for {names}")
                return None
            try:
                name, data = self.q.get(timeout=left)
            except self.queue.Empty:
                continue
            if name in names:
                return name, data


def _self_test():
    import requests
    import tempfile
    global HUMAN_STATS_PATH, THINK_DELAY, LLM_DAILY_CAP, LLM_ENABLED
    HUMAN_STATS_PATH = os.path.join(tempfile.mkdtemp(), "human_stats.json")
    THINK_DELAY = 0.05
    url = _start_test_server(5057)

    print("== REST ==")
    assert requests.get(url + "/api/health").json() == {"ok": True}
    d = requests.get(url + "/api/difficulties").json()
    assert [x["id"] for x in d] == ["easy", "medium", "hard"]
    print("health OK | difficulties:", [(x["id"], x["checkpoint"]) for x in d])
    lb = requests.get(url + "/api/leaderboard")
    print("leaderboard status:", lb.status_code)
    if lb.status_code == 200:
        for p in lb.json()["players"]:
            print(f"  {p['name']:24s} games={p['games']:4d} win_rate={p['win_rate']}")
    mt = requests.get(url + "/api/metrics")
    print("metrics status:", mt.status_code)
    if mt.status_code == 200:
        j = mt.json()
        print("  history points:", len(j["history"]), "| finished:", j.get("finished"))

    print()
    print("== Human vs AI over a real socket (hard) ==")
    c = _TestClient(url)
    c.send("new_game", {"difficulty": "hard", "human_first": True})
    _, st = c.wait(["game_state"])
    assert st["current_player"] == st["human"] and not st["done"]
    c.send("human_move", {"col": 9})
    _, err = c.wait(["error"])
    print("bad column answer:", err["message"])
    prefer = [3, 2, 4, 1, 5, 0, 6]
    first_ai = None
    n = 0
    while True:
        col = next(x for x in prefer if x in st["legal"])
        c.send("human_move", {"col": col})
        name, data = c.wait(["ai_move", "game_over", "error"])
        assert name != "error", data
        n += 1
        if name == "ai_move":
            first_ai = first_ai or data
            st = data["state"]
            if st["done"]:
                _, over = c.wait(["game_over"])
                break
        else:
            over = data
            break
    b = first_ai["brain"]
    assert len(b["q_values"]) == 7 and len(b["input"]) == 84
    assert len(b["hidden1"]) == 12 and len(b["hidden2"]) == 12
    print(f"game over after {n} human moves | winner_who={over['winner_who']} "
          f"| winning cells={len(over['winning_cells'])}")
    stats = read_human_stats()
    assert sum(stats.values()) == 1
    print("human stats saved:", stats)

    c.send("new_game", {"difficulty": "easy", "human_first": False})
    _, ev = c.wait(["ai_move"])
    assert ev["who"] == "ai"
    print("AI-first game: AI opened in column", ev["column"])
    assert sum(read_human_stats().values()) == 1
    print("abandoned game was not counted: OK")

    print()
    print("== Spectator match: AI vs Minimax depth 2, speed 20 ==")
    for ai_first in (True, False):
        c.send("start_match", {"opponent": "minimax2", "ai_first": ai_first,
                               "speed": 20, "difficulty": "hard"})
        c.wait(["game_state"])
        whos = []
        while True:
            name, data = c.wait(["ai_move", "game_over"], timeout=30)
            if name == "game_over":
                break
            whos.append(data["who"])
        print(f"ai_first={ai_first!s:5s}: moves ai={whos.count('ai')} "
              f"minimax={whos.count('minimax')} | winner_who={data['winner_who']}")

    print()
    print("== stop_match ==")
    THINK_DELAY = 0.6
    c.send("start_match", {"opponent": "minimax2", "ai_first": True, "speed": 0.25})
    c.wait(["ai_thinking"])
    c.send("stop_match")
    c.wait(["match_stopped"])
    c.drain()
    late = c.wait(["ai_move"], timeout=3.5, required=False)
    assert late is None, "a move arrived after stop_match"
    print("no moves after stop: OK")

    print()
    print("== LLM limits (no real LLM call is made) ==")
    LLM_DAILY_CAP = 2
    _llm_day["day"], _llm_day["used"] = None, 0
    spent = [llm_try_spend() for _ in range(3)]
    assert spent == [True, True, False], spent
    assert llm_remaining() == 0
    print("daily cap of 2: the third LLM move is refused: OK")

    LLM_DAILY_CAP = 0
    c.drain()
    c.send("start_match", {"opponent": "llm", "speed": 20})
    _, err = c.wait(["error"])
    assert "daily limit" in err["message"], err
    assert c.wait(["game_state", "ai_move"], timeout=1.5, required=False) is None
    print("cap reached, match refused:", err["message"][:58] + "...")

    LLM_DAILY_CAP = 200
    LLM_ENABLED = False
    c.drain()
    c.send("start_match", {"opponent": "llm", "speed": 20})
    _, err = c.wait(["error"])
    assert "switched off" in err["message"], err
    print("LLM switched off, match refused:", err["message"][:58] + "...")
    LLM_ENABLED = True

    c.sio.disconnect()
    print("Self-test finished.")


def _llm_test():
    global THINK_DELAY, HUMAN_STATS_PATH
    import tempfile
    HUMAN_STATS_PATH = os.path.join(tempfile.mkdtemp(), "human_stats.json")
    THINK_DELAY = 0.05
    url = _start_test_server(5058)
    c = _TestClient(url)
    print("Starting AI vs LLM. The first LLM move can take 10+ seconds...")
    t0 = time.time()
    c.send("start_match", {"opponent": "llm", "ai_first": True, "speed": 20})
    c.wait(["game_state", "error"])
    while True:
        name, data = c.wait(["ai_thinking", "ai_move", "error", "game_over"], timeout=120)
        if name == "error":
            print("ERROR:", data)
            break
        if name == "ai_thinking" and data["who"] == "llm":
            print("saw the 'LLM is thinking' event:", data.get("message"))
        if name == "ai_move" and data["who"] == "llm":
            info = dict(data["llm"] or {})
            if "raw" in info:
                info["raw"] = str(info["raw"])[:80]
            print(f"LLM moved after {time.time() - t0:.1f}s: column={data['column']}")
            print("llm info:", info)
            break
        if name == "game_over":
            print("game ended before the LLM moved")
            break
    c.send("stop_match")
    c.sio.disconnect()


if __name__ == "__main__":
    if "--test" in sys.argv:
        _self_test()
    elif "--llm-test" in sys.argv:
        _llm_test()
    else:
        port = int(os.environ.get("PORT", "5000"))
        print(f"Connect Four AI Arena backend on http://localhost:{port}")
        socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)
