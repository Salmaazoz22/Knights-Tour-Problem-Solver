import time
import csv

# ----------------------------
# IMPORT YOUR SOLVERS
# ----------------------------
# Backtracking
from core.algorithms.backtracking.brute_force import BruteForceBacktracking
from core.algorithms.backtracking.randomized_heuristic import BacktrackingSolver
from core.algorithms.backtracking.warnsdorff import WarnsdorffBacktracking

# GA Solvers
from core.algorithms.genetic import GeneticAlgorithmClassic as GAClassic
from core.algorithms.genetic import GeneticAlgorithmOptimized as GAOptimized
from core.algorithms.genetic import GeneticAlgorithmElitism as GAElitism

# Cultural Algorithm (optional)
try:
    from core.algorithms.cultural.cultural_algorithm import CulturalAlgorithm
    CA_AVAILABLE = True
except ImportError:
    print("⚠️ Cultural Algorithm not available")
    CA_AVAILABLE = False

# Import Board
from core.board import Board

# ----------------------------
# WRAPPERS FOR CONSISTENT SIGNATURE
# ----------------------------
class BacktrackingWrapper:
    def __init__(self, solver_class, board_size, start_pos):
        # Backtracking solvers expect a Board object
        board = Board(board_size)
        self.solver = solver_class(board)
        self.start_pos = start_pos
        self.board_size = board_size
    
    def run(self):
        # Call solve() method with start_pos
        result = self.solver.solve(self.start_pos)
        # Create a wrapper object that mimics Individual with fitness_value
        class ResultWrapper:
            def __init__(self, result, board_size):
                if result is not None:
                    self.fitness_value = board_size * board_size - 1  # Complete tour
                    self.fitness = self.fitness_value
                else:
                    self.fitness_value = 0
                    self.fitness = 0
        return ResultWrapper(result, self.board_size)

class GAWrapper:
    def __init__(self, solver_class, board_size, start_pos):
        # Convert linear index to row, col
        start_row = start_pos // board_size
        start_col = start_pos % board_size
        self.solver = solver_class(board_size=board_size, start_row=start_row, start_col=start_col)
    
    def run(self):
        return self.solver.run()

class CAWrapper:
    def __init__(self, solver_class, board_size, start_pos):
        self.solver = solver_class(board_size=board_size, start_pos=start_pos, pop_size=150, max_gen=1000)
    
    def run(self):
        return self.solver.run()

# ----------------------------
# SOLVER LISTS
# ----------------------------
BACKTRACKING_SOLVERS = [
    ("Brute Force Backtracking", BruteForceBacktracking),
    ("Randomized Heuristic Backtracking", BacktrackingSolver),
    ("Warnsdorff Backtracking", WarnsdorffBacktracking),
]

GA_SOLVERS = [
    ("GA Classic", GAClassic),
    ("GA Optimized", GAOptimized),
    ("GA Elitism", GAElitism),
]

if CA_AVAILABLE:
    GA_SOLVERS.append(("Cultural Algorithm", CulturalAlgorithm))

# ----------------------------
# PARAMETERS
# ----------------------------
BOARD_SIZES = [5, 6, 7, 8,9,10]  
START_POSITIONS = [(0,0), (0,-1), (-1,0), (-1,-1), (2,2)]
RUNS_PER_SETTING = 3  # Reduced for faster testing

# ----------------------------
# HELPER FUNCTION TO RUN SOLVERS
# ----------------------------
def run_solver(name, solver_class, size, pos, runs=RUNS_PER_SETTING):
    """Run a solver multiple times and collect statistics."""
    # Convert position to linear index
    r = pos[0] if pos[0] >= 0 else size + pos[0]
    c = pos[1] if pos[1] >= 0 else size + pos[1]
    start_square = r * size + c

    max_fitness = size * size - 1
    successful = 0
    total_time = 0
    best_fit = 0

    print(f"\nTesting {name} on {size}x{size} board, start=({r},{c})...")

    for run_num in range(runs):
        print(f"  Run {run_num + 1}/{runs}...", end=" ")
        start_time = time.time()

        try:
            # Use wrappers to unify constructors
            if "Cultural" in name:
                solver = CAWrapper(solver_class, size, start_square)
            elif "GA" in name:
                solver = GAWrapper(solver_class, size, start_square)
            else:
                solver = BacktrackingWrapper(solver_class, size, start_square)

            best_ind = solver.run()
            
            # Get fitness value
            f = getattr(best_ind, "fitness_value", getattr(best_ind, "fitness", 0))

            runtime = time.time() - start_time
            total_time += runtime
            best_fit = max(best_fit, f)
            
            if f == max_fitness:
                successful += 1
                print(f"✓ Success in {runtime:.2f}s")
            else:
                print(f"✗ Fitness {f}/{max_fitness} in {runtime:.2f}s")
                
        except Exception as e:
            print(f"✗ Error: {e}")
            runtime = time.time() - start_time
            total_time += runtime

    return {
        "Algorithm": name,
        "Size": size,
        "Start_R": r,
        "Start_C": c,
        "Success_Rate_%": (successful / runs) * 100,
        "Avg_Runtime_s": total_time / runs,
        "Best_Fitness": best_fit,
        "Max_Fitness": max_fitness,
        "Runs": runs
    }

# ----------------------------
# MAIN LOOP
# ----------------------------
if __name__ == "__main__":
    print("="*70)
    print("KNIGHT'S TOUR SOLVER COMPARISON TEST")
    print("="*70)
    
    all_results = []

    # Test Backtracking solvers first (smaller boards)
    print("\n" + "="*70)
    print("TESTING BACKTRACKING ALGORITHMS")
    print("="*70)
    
    for size in [5, 6]:  # Only test small boards for backtracking
        for pos in START_POSITIONS:
            # Skip invalid positions
            r_test = pos[0] if pos[0] >= 0 else size + pos[0]
            c_test = pos[1] if pos[1] >= 0 else size + pos[1]
            if r_test >= size or c_test >= size or r_test < 0 or c_test < 0:
                continue
                
            for name, cls in BACKTRACKING_SOLVERS:
                result = run_solver(name, cls, size, pos)
                all_results.append(result)

    # Test GA + CA solvers (all board sizes)
    print("\n" + "="*70)
    print("TESTING GENETIC & CULTURAL ALGORITHMS")
    print("="*70)
    
    for size in BOARD_SIZES:
        for pos in START_POSITIONS:
            # Skip invalid positions
            r_test = pos[0] if pos[0] >= 0 else size + pos[0]
            c_test = pos[1] if pos[1] >= 0 else size + pos[1]
            if r_test >= size or c_test >= size or r_test < 0 or c_test < 0:
                continue
                
            for name, cls in GA_SOLVERS:
                result = run_solver(name, cls, size, pos)
                all_results.append(result)

    # Save to CSV
    csv_file = "all_solvers_comparison.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "="*70)
    print(f"✔ All solvers comparison saved → {csv_file}")
    print("="*70)
    
    # Print summary
    print("\nSUMMARY:")
    print("-"*70)
    for result in all_results:
        print(f"{result['Algorithm']:40s} | Size {result['Size']} | "
              f"Success: {result['Success_Rate_%']:5.1f}% | "
              f"Avg Time: {result['Avg_Runtime_s']:6.2f}s")