"""
Connect Four game engine.

Board is 6 rows x 7 columns. Row 0 is the TOP, row 5 is the BOTTOM.
Players are 1 and -1. Player 1 always moves first.
Run this file directly to run the self-tests.
"""
import numpy as np

ROWS = 6
COLS = 7


class Connect4:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = 1   # whose turn it is (1 or -1)
        self.winner = 0           # 0 = nobody, 1 or -1 = that player won
        self.done = False         # True when someone wins or the board is full
        self.winning_cells = []   # (row, col) cells of the winning line
        self.move_count = 0
        self.last_move = None     # (row, col) of the last disc dropped
        return self.get_state()

    def copy(self):
        # Fast copy, used later by minimax to try moves without touching the real game
        other = Connect4.__new__(Connect4)
        other.board = self.board.copy()
        other.current_player = self.current_player
        other.winner = self.winner
        other.done = self.done
        other.winning_cells = list(self.winning_cells)
        other.move_count = self.move_count
        other.last_move = self.last_move
        return other

    def legal_moves(self):
        # A column is playable if its top cell is empty
        if self.done:
            return []
        return [c for c in range(COLS) if self.board[0, c] == 0]

    def legal_mask(self):
        # 7 numbers: 1.0 for playable columns, 0.0 for full ones (used for illegal-move masking)
        mask = np.zeros(COLS, dtype=np.float32)
        for c in self.legal_moves():
            mask[c] = 1.0
        return mask

    def is_legal(self, col):
        return col in self.legal_moves()

    def step(self, col):
        """Drop a disc for the current player. Returns True if the game is over."""
        if not self.is_legal(col):
            raise ValueError(f"Illegal move: column {col}")

        # Gravity: the disc lands on the lowest empty row of the column
        row = ROWS - 1
        while self.board[row, col] != 0:
            row -= 1

        self.board[row, col] = self.current_player
        self.move_count += 1
        self.last_move = (row, col)

        cells = self._find_win(row, col)
        if cells:
            self.winner = self.current_player
            self.winning_cells = cells
            self.done = True
        elif self.move_count == ROWS * COLS:
            self.done = True      # board full, nobody won = draw
        else:
            self.current_player = -self.current_player
        return self.done

    def _find_win(self, row, col):
        # Only the disc just dropped can create a new win, so check lines through it.
        player = int(self.board[row, col])
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):  # right, down, two diagonals
            cells = [(row, col)]
            for sign in (1, -1):  # walk both ways along the line
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
                    cells.append((r, c))
                    r += sign * dr
                    c += sign * dc
            if len(cells) >= 4:
                return sorted(cells)
        return []

    def is_draw(self):
        return self.done and self.winner == 0

    def get_state(self):
        # 2 planes of 6x7, seen from the view of the player whose turn it is:
        #   plane 0 = "my discs", plane 1 = "their discs", then flattened to 84 numbers
        me = (self.board == self.current_player).astype(np.float32)
        them = (self.board == -self.current_player).astype(np.float32)
        return np.concatenate([me.flatten(), them.flatten()])

    def render(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        lines = [" ".join(symbols[int(v)] for v in row) for row in self.board]
        lines.append(" ".join(str(c) for c in range(COLS)))
        return "\n".join(lines)


def play(moves):
    game = Connect4()
    for m in moves:
        game.step(m)
    return game


def run_tests():
    # 1. Vertical win
    g = play([0, 1, 0, 1, 0, 1, 0])
    assert g.done and g.winner == 1 and len(g.winning_cells) == 4, "vertical win failed"
    print("PASS vertical win")

    # 2. Horizontal win
    g = play([0, 0, 1, 1, 2, 2, 3])
    assert g.done and g.winner == 1 and len(g.winning_cells) == 4, "horizontal win failed"
    print("PASS horizontal win")

    # 3. Diagonal wins (both directions)
    g = play([0, 1, 1, 2, 3, 2, 2, 3, 6, 3, 3])
    assert g.done and g.winner == 1 and len(g.winning_cells) == 4, "diagonal win failed"
    g = play([6, 5, 5, 4, 3, 4, 4, 3, 0, 3, 3])
    assert g.done and g.winner == 1 and len(g.winning_cells) == 4, "anti-diagonal win failed"
    print("PASS both diagonal wins")

    # 4. Draw: a full board with no four in a row
    draw_moves = [0, 2, 2, 0] * 3 + [1, 3, 3, 1] * 3 + [4, 6, 6, 4] * 3 + [5] * 6
    g = play(draw_moves)
    assert g.done and g.winner == 0 and g.is_draw() and g.legal_moves() == [], "draw failed"
    print("PASS draw detection")

    # 5. Full column is illegal
    g = play([0] * 6)
    assert g.legal_moves() == [1, 2, 3, 4, 5, 6], "full column should not be legal"
    try:
        g.step(0)
        raise AssertionError("illegal move was allowed")
    except ValueError:
        pass
    print("PASS illegal moves blocked")

    # 6. State: 84 numbers, always from the view of the player to move
    g = Connect4()
    assert g.get_state().shape == (84,) and g.get_state().sum() == 0, "empty state wrong"
    g.step(3)  # player 1 drops in column 3, now it is player -1's turn
    s = g.get_state()
    assert s.sum() == 1 and s[42 + 5 * 7 + 3] == 1, "state perspective wrong"
    print("PASS state encoding")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()
    demo = play([3, 3, 4, 2, 5])
    print("\nSample board (X = player 1, O = player -1):")
    print(demo.render())
