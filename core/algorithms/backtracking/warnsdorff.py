# -*- coding: utf-8 -*-
"""
Warnsdorff's Rule (Classic) for Knight's Tour
Pure greedy algorithm - no backtracking
Fast but may get stuck on larger boards
"""

from core.board import Board


class WarnsdorffBacktracking:
    """
    Classic Warnsdorff's heuristic (greedy, no backtracking).
    Always picks the move with fewest onward options.
    Very fast but may fail on some starting positions.
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
        Solve using Warnsdorff's heuristic (greedy).
        
        Args:
            start_square: Starting square index
            callback: Optional callback(path, step)
        
        Returns:
            list: Solution path or None
        """
        self.callback = callback
        self.running = True
        
        # Convert to (row, col)
        x = start_square // self.size
        y = start_square % self.size
        
        # Initialize
        self.grid_visited = [[-1 for _ in range(self.size)] for _ in range(self.size)]
        self.grid_visited[x][y] = 0
        self.solution = [(x, y)]
        
        # Greedy algorithm (no backtracking)
        for move_number in range(1, self.total_squares):
            if not self.running:
                return None
            
            # Callback
            if self.callback:
                path_indices = [r * self.size + c for r, c in self.solution]
                self.callback(path_indices, move_number)
            
            # Get legal moves
            legal_moves = self._get_moves(x, y)
            
            if not legal_moves:
                return None  # Stuck - no solution
            
            # Sort by Warnsdorff (lowest degree first)
            legal_moves = self._sorted_moves(legal_moves)
            nx, ny = legal_moves[0]  # Pick best
            
            # Make move
            x, y = nx, ny
            self.grid_visited[x][y] = move_number
            self.solution.append((x, y))
        
        # Success
        return [r * self.size + c for r, c in self.solution]
    
    def _is_valid(self, x, y):
        """Check if (x, y) is valid and unvisited."""
        return (0 <= x < self.size and 
                0 <= y < self.size and 
                self.grid_visited[x][y] == -1)
    
    def _get_moves(self, x, y):
        """Get all valid moves from (x, y)."""
        moves = []
        for dx, dy in self.MOVES:
            nx, ny = x + dx, y + dy
            if self._is_valid(nx, ny):
                moves.append((nx, ny))
        return moves
    
    def _degree(self, x, y):
        """Count onward moves from (x, y)."""
        count = 0
        for dx, dy in self.MOVES:
            nx, ny = x + dx, y + dy
            if self._is_valid(nx, ny):
                count += 1
        return count
    
    def _sorted_moves(self, moves):
        """Sort moves by Warnsdorff degree (ascending)."""
        return sorted(moves, key=lambda m: self._degree(m[0], m[1]))
    
    def stop(self):
        """Stop the search."""
        self.running = False
# =======================
# Main runner
# =======================
if __name__ == "__main__":
    print("=== Knight's Tour | Warnsdorff Greedy (Classic) ===")

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

    # ---- Create board and solver ----
    board = Board(size=size)
    solver = WarnsdorffBacktracking(board)

    print("\n⏳ Solving... (Greedy, no backtracking)")

    solution = solver.solve(start_square)

    # ---- Output ----
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
        print("\n❌ No solution found (stuck position)")
