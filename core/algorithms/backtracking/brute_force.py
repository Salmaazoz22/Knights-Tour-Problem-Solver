# -*- coding: utf-8 -*-
"""
Brute Force Backtracking for Knight's Tour
Pure backtracking without any heuristics
WARNING: Very slow, use only for small boards (5×5 or 6×6)
"""

from core.board import Board


class BruteForceBacktracking:
    """
    Pure brute force backtracking.
    Tries all possible moves without optimization.
    """

    MOVES = [
        (+2, +1), (+1, +2), (-1, +2), (-2, +1),
        (-2, -1), (-1, -2), (+1, -2), (+2, -1)
    ]

    def __init__(self, board):
        self.board = board
        self.size = board.size
        self.total_squares = board.num_squares
        self.solution = []
        self.callback = None
        self.running = True
        self.grid_visited = None

    def solve(self, start_square, callback=None):
        """
        Solve using brute force.
        """
        self.callback = callback
        self.running = True

        start_x = start_square // self.size
        start_y = start_square % self.size

        self.grid_visited = [[-1 for _ in range(self.size)] for _ in range(self.size)]
        self.grid_visited[start_x][start_y] = 0
        self.solution = [(start_x, start_y)]

        if self._backtrack(start_x, start_y, 1):
            return [r * self.size + c for r, c in self.solution]
        return None

    def _is_valid(self, x, y):
        return (
            0 <= x < self.size and
            0 <= y < self.size and
            self.grid_visited[x][y] == -1
        )

    def _backtrack(self, x, y, step):
        if not self.running:
            return False

        if step == self.total_squares:
            return True

        if self.callback:
            path_indices = [r * self.size + c for r, c in self.solution]
            self.callback(path_indices, step)

        for dx, dy in self.MOVES:
            nx, ny = x + dx, y + dy

            if self._is_valid(nx, ny):
                self.grid_visited[nx][ny] = step
                self.solution.append((nx, ny))

                if self._backtrack(nx, ny, step + 1):
                    return True

                self.grid_visited[nx][ny] = -1
                self.solution.pop()

        return False

    def stop(self):
        self.running = False


# =======================
# Main runner
# =======================
if __name__ == "__main__":
    print("=== Knight's Tour | Brute Force Backtracking ===")

    # ---- Board size (restricted) ----
    while True:
        try:
            size = int(input("Enter board size (5 or 6 only): "))
            if size in (5, 6):
                break
            print("⚠️ Brute force is too slow for sizes > 6")
        except ValueError:
            print("❌ Please enter a valid number")

    # ---- Start position ----
    while True:
        try:
            row = int(input(f"Enter start row (0 - {size - 1}): "))
            col = int(input(f"Enter start col (0 - {size - 1}): "))
            if 0 <= row < size and 0 <= col < size:
                break
            print("⚠️ Invalid position")
        except ValueError:
            print("❌ Please enter valid numbers")

    start_square = row * size + col

    board = Board(size=size)
    solver = BruteForceBacktracking(board)

    print("\n⏳ Solving... (Brute Force may be VERY slow)")

    solution = solver.solve(start_square)

    if solution:
        print("\n✅ Solution found!")
        print("Path (indices):")
        print(solution)

        print("\nBoard order:")
        board_view = [[-1 for _ in range(size)] for _ in range(size)]
        for step, idx in enumerate(solution):
            r = idx // size
            c = idx % size
            board_view[r][c] = step

        for r in board_view:
            print(" ".join(f"{v:2}" for v in r))
    else:
        print("\n❌ No solution found (or search stopped)")
