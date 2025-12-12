import random
import copy
import time
import math

class Board:
    def __init__(self, size=8):
        self.size = size
        self.num_squares = size * size
        self.valid_moves = self._precompute_valid_moves()

    def _precompute_valid_moves(self):
        deltas = [(2, 1), (1, 2), (-1, 2), (-2, 1),
                  (-2, -1), (-1, -2), (1, -2), (2, -1)]
        moves = {}
        for square in range(self.num_squares):
            row, col = divmod(square, self.size)
            neighbors = []
            for dr, dc in deltas:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < self.size and 0 <= new_col < self.size:
                    neighbors.append(new_row * self.size + new_col)
            moves[square] = neighbors
        return moves

    def is_valid_move(self, from_sq, to_sq):
        return to_sq in self.valid_moves[from_sq]

class Individual:
    def __init__(self, board, start_pos, chromosome=None):
        self.board = board
        self.start_pos = start_pos
        self.num_squares = board.num_squares
        
        if chromosome is None:
            # --- GREEDY RANDOM INITIALIZATION ---
            # (No Warnsdorff / No Look-ahead)
            self.chromosome = [start_pos]
            visited = {start_pos}
            current = start_pos
            
            # Loop to fill the rest of the board
            for _ in range(self.num_squares - 1):
                # 1. Get all valid moves from 'current' that are unvisited
                possible_moves = self.board.valid_moves[current]
                unvisited_moves = [m for m in possible_moves if m not in visited]
                
                if unvisited_moves:
                    # 2. Pick a random valid neighbor (Greedy but blind)
                    next_move = random.choice(unvisited_moves)
                else:
                    # 3. If stuck (dead end), jump to any random unvisited square
                    remaining = [sq for sq in range(self.num_squares) if sq not in visited]
                    next_move = random.choice(remaining)
                
                self.chromosome.append(next_move)
                visited.add(next_move)
                current = next_move
        else:
            self.chromosome = chromosome[:]
        
        self.fitness_value = self._calculate_fitness_value()

    def _calculate_fitness_value(self):
        valid_count = 0
        for i in range(len(self.chromosome) - 1):
            if self.board.is_valid_move(self.chromosome[i], self.chromosome[i + 1]):
                valid_count += 1
        return valid_count

    def mutate(self, mutation_prob=0.05):
        # --- INVERSION MUTATION ---
        # Reverses a segment to fix broken paths without destroying connections
        if random.random() < mutation_prob:
            i = random.randint(1, len(self.chromosome) - 2)
            j = random.randint(i + 1, len(self.chromosome) - 1)
            self.chromosome[i:j+1] = self.chromosome[i:j+1][::-1]
            self.fitness_value = self._calculate_fitness_value()

    def get_fitness(self):
        return self.fitness_value

    def __repr__(self):
        return f"Individual(fitness={self.fitness_value})"

class GeneticAlgorithm:
    def __init__(self, board_size=8, start_row=0, start_col=0, pop_size=150, generations=500,
                 cx_prob=0.7, mut_prob=0.3, elite_ratio=0.1):
        self.board = Board(board_size)
        self.start_pos = start_row * board_size + start_col
        self.pop_size = pop_size
        self.generations = generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.elite_ratio = elite_ratio
        
        # Population is just a list of Individuals (No separate class)
        print("Initializing population with Greedy Random strategy...")
        self.population = [Individual(self.board, self.start_pos) for _ in range(pop_size)]

    def get_best_individual(self):
        return max(self.population, key=lambda ind: ind.get_fitness())

    def get_population_stats(self):
        total_fitness = sum(ind.get_fitness() for ind in self.population)
        avg_fitness = total_fitness / self.pop_size
        return total_fitness, avg_fitness

    def select_parent(self, tournament_size=5):
        candidates = random.sample(self.population, tournament_size)
        return max(candidates, key=lambda ind: ind.get_fitness())

    def _crossover(self, chromo1, chromo2):
        """Robust Order Crossover (OX1)."""
        size = len(chromo1)
        cx1 = random.randint(1, size - 1) 
        cx2 = random.randint(cx1, size - 1)
        
        def create_child(p1, p2):
            child = [-1] * size
            child[0] = p1[0] # Fixed start
            
            child[cx1:cx2] = p1[cx1:cx2]
            
            existing = set(child[cx1:cx2])
            existing.add(child[0])
            
            # Fill remaining from p2 (wrapping around)
            candidates = p2[cx2:] + p2[:cx2]
            needed = [gene for gene in candidates if gene not in existing]
            
            fill_indices = list(range(cx2, size)) + list(range(1, cx1))
            
            for i, gene in zip(fill_indices, needed):
                child[i] = gene
            return child

        child1 = create_child(chromo1, chromo2)
        child2 = create_child(chromo2, chromo1)
        return child1, child2

    def evolve(self):
        new_pop = []
        
        # 1. Elitism
        num_elites = int(self.pop_size * self.elite_ratio)
        sorted_inds = sorted(self.population, key=lambda ind: ind.get_fitness(), reverse=True)
        new_pop.extend(copy.deepcopy(sorted_inds[:num_elites]))

        # 2. Crossover & Mutation
        while len(new_pop) < self.pop_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            
            if random.random() < self.cx_prob:
                c1_chromo, c2_chromo = self._crossover(parent1.chromosome, parent2.chromosome)
                child1 = Individual(self.board, self.start_pos, c1_chromo)
                child2 = Individual(self.board, self.start_pos, c2_chromo)
            else:
                child1 = copy.deepcopy(parent1)
                child2 = copy.deepcopy(parent2)

            if random.random() < self.mut_prob:
                child1.mutate(self.mut_prob)
            if random.random() < self.mut_prob:
                child2.mutate(self.mut_prob)

            if len(new_pop) < self.pop_size:
                new_pop.append(child1)
            if len(new_pop) < self.pop_size:
                new_pop.append(child2)

        self.population = new_pop[:self.pop_size]

    def run(self):
        start_time = time.time()
        
        max_fitness_so_far = -1
        stagnation_counter = 0
        stagnation_limit = 500

        print(f"\n{'Gen':<5} | {'Best':<5} | {'Avg':<6} | {'Total':<8} | {'Stagnation':<10} | {'Status'}")
        print("-" * 65)

        for gen in range(self.generations):
            total_fit, avg_fit = self.get_population_stats()
            best_ind = self.get_best_individual()
            best_fit = best_ind.get_fitness()

            if best_fit > max_fitness_so_far:
                max_fitness_so_far = best_fit
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            print(f"{gen:<5} | {best_fit:<5} | {avg_fit:<6.2f} | {total_fit:<8} | {stagnation_counter:<10} | Evolving...", end='\r')

            # Check Solution
            if best_fit == self.board.num_squares - 1:
                print(f"{gen:<5} | {best_fit:<5} | {avg_fit:<6.2f} | {total_fit:<8} | {stagnation_counter:<10} | FOUND!      ")
                print(f"\nFull tour found in {time.time() - start_time:.2f}s")
                return best_ind

            # Check Stagnation
            if stagnation_counter >= stagnation_limit:
                print(f"{gen:<5} | {best_fit:<5} | {avg_fit:<6.2f} | {total_fit:<8} | {stagnation_counter:<10} | STAGNATED   ")
                print(f"\nStopping: No improvement for {stagnation_limit} generations.")
                return best_ind

            self.evolve()
            print(f"{gen:<5} | {best_fit:<5} | {avg_fit:<6.2f} | {total_fit:<8} | {stagnation_counter:<10} |             ")

        print(f"\nCompleted max generations in {time.time() - start_time:.2f} seconds.")
        return self.get_best_individual()

# --------------------------
# MAIN PROGRAM
# --------------------------
if __name__ == "__main__":
    random.seed(42)
    
    # Input 1: Board Size
    while True:
        try:
            sz_input = input("Enter board size (5-10): ")
            board_size = int(sz_input)
            if 5 <= board_size <= 10:
                break
            print("Size must be between 5 and 10.")
        except ValueError:
            print("Invalid input.")

    # Input 2: Start Position
    while True:
        try:
            pos_input = input(f"Enter starting position (row col) [0-{board_size-1}]: ")
            parts = pos_input.split()
            if len(parts) == 2:
                start_row, start_col = int(parts[0]), int(parts[1])
                if 0 <= start_row < board_size and 0 <= start_col < board_size:
                    break
            print(f"Coordinates must be between 0 and {board_size-1}.")
        except ValueError:
            print("Invalid input.")

    print(f"\nInitializing GA for {board_size}x{board_size} board starting at ({start_row}, {start_col})...")
    
    ga = GeneticAlgorithm(
        board_size=board_size, 
        start_row=start_row, 
        start_col=start_col, 
        pop_size=200, 
        generations=2000
    )
    
    best = ga.run()
    
    print("\nBest tour found (Linear Indices):", best.chromosome)
    
    # Visual check
    grid = [[-1] * board_size for _ in range(board_size)]
    for i, idx in enumerate(best.chromosome):
        r, c = divmod(idx, board_size)
        grid[r][c] = i
    
    print("\nGrid Visualization (Move Order):")
    for row in grid:
        print("\t".join(str(x) for x in row))