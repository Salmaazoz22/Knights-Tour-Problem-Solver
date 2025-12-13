# -*- coding: utf-8 -*-
"""
Backtracking algorithm with Randomized Heuristic for Knight's Tour.
Uses Warnsdorff's heuristic with randomization for equal-degree moves.
"""

import random
from core.board import Board


class BacktrackingSolver:
    """
    Backtracking algorithm with Randomized Warnsdorff's heuristic.
    """

    MOVES = [
        (+2, +1), (+1, +2), (-1, +2), (-2, +1),
        (-2, -1), (-1, -2), (+1, -2), (+2, -1)
    ]

    def __init__(self, board):
        self.board = board
        self.solution = []
        self.callback = None
        self.running = True
        self.visited = None

    def solve(self, start_square, callback=None):
        self.callback = callback
        self.solution = [start_square]
        self.running = True

        self.visited = [False] * self.board.num_squares
        self.visited[start_square] = True

        if self._backtrack_randomized(start_square, 1):
            return self.solution
        return None

    def _backtrack_randomized(self, current_sq, move_count):
        if not self.running:
            return False

        if move_count == self.board.num_squares:
            return True

        if self.callback:
            self.callback(self.solution[:], move_count)

        candidates = [
            sq for sq in self.board.valid_moves[current_sq]
            if not self.visited[sq]
        ]

        if not candidates:
            return False

        candidates = self._sort_moves_randomized(candidates)

        for next_sq in candidates:
            self.visited[next_sq] = True
            self.solution.append(next_sq)

            if self._backtrack_randomized(next_sq, move_count + 1):
                return True

            self.visited[next_sq] = False
            self.solution.pop()

            if self.callback:
                self.callback(self.solution[:], move_count)

        return False

    def _sort_moves_randomized(self, candidates):
        degree_groups = {}

        for sq in candidates:
            degree = self._count_onward_moves(sq)
            degree_groups.setdefault(degree, []).append(sq)

        sorted_candidates = []
        for degree in sorted(degree_groups.keys()):
            group = degree_groups[degree]
            random.shuffle(group)
            sorted_candidates.extend(group)

        return sorted_candidates

    def _count_onward_moves(self, square):
        return sum(
            1 for sq in self.board.valid_moves[square]
            if not self.visited[sq]
        )

    def stop(self):
        self.running = False


class BacktrackingSolverClassic:
    """
    Classic Backtracking using pure Warnsdorff heuristic (no randomization).
    """

    def __init__(self, board):
        self.board = board
        self.solution = []
        self.callback = None
        self.running = True

    def solve(self, start_square, callback=None):
        self.callback = callback
        self.solution = [start_square]
        self.running = True

        visited = [False] * self.board.num_squares
        visited[start_square] = True

        if self._backtrack(start_square, visited, 1):
            return self.solution
        return None

    def _backtrack(self, current_sq, visited, move_count):
        if not self.running:
            return False

        if move_count == self.board.num_squares:
            return True

        if self.callback:
            self.callback(self.solution[:], move_count)

        candidates = [
            sq for sq in self.board.valid_moves[current_sq]
            if not visited[sq]
        ]

        if not candidates:
            return False

        candidates.sort(
            key=lambda sq: sum(
                1 for s in self.board.valid_moves[sq]
                if not visited[s]
            )
        )

        for next_sq in candidates:
            visited[next_sq] = True
            self.solution.append(next_sq)

            if self._backtrack(next_sq, visited, move_count + 1):
                return True

            visited[next_sq] = False
            self.solution.pop()

            if self.callback:
                self.callback(self.solution[:], move_count)

        return False

    def stop(self):
        self.running = False


# =======================
# Main runner
# =======================
if __name__ == "__main__":
    print("=== Knight's Tour | Backtracking (Warnsdorff + Randomization) ===")

    # ---- Board size ----
    while True:
        try:
            size = int(input("Enter board size (5 - 10): "))
            if 5 <= size <= 10:
                break
            print("⚠️ Size must be between 5 and 10")
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

    print("\nChoose algorithm:")
    print("1 - Randomized Warnsdorff Backtracking")
    print("2 - Classic Warnsdorff Backtracking")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "2":
        solver = BacktrackingSolverClassic(board)
        print("\n▶ Running Classic Warnsdorff...")
    else:
        solver = BacktrackingSolver(board)
        print("\n▶ Running Randomized Warnsdorff...")

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
        print("\n❌ No solution found")
