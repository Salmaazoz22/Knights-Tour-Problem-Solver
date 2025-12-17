# Knight's Tour Problem Solver 🐴♟️

A comprehensive solution to the **Knight's Tour Problem** using **Cultural Algorithm** and **Backtracking** with an interactive GUI.

## 👥 Team Members

- **Salma Mohamed Abdelaziz**
- **Rawan Essam-ElDin Fahmy**
- **Rowayda Mohammed Khairy**
- **Salma Ashraf**
- **Zeiad Saher Salama**
- **Mohammed Saied Ahmed**

---

## 📋 Project Overview

The Knight's Tour is a classic chess problem where a knight must visit every square on a chessboard exactly once. This project implements two powerful approaches to solve this challenging problem:

1. **Cultural Algorithm (CA)** - An evolutionary approach combining population-based search with knowledge-based belief space
2. **Backtracking with Warnsdorff's Heuristic** - A systematic search algorithm optimized with intelligent move selection

This project was developed as part of the **AI310 & CS361 Artificial Intelligence** course at **Helwan University, Faculty of Computing & Artificial Intelligence**.

---

## 🚀 Features

✅ **Interactive GUI** with real-time chessboard visualization  
✅ **Two Powerful Algorithms** (Cultural Algorithm & Backtracking)  
✅ **Adjustable Board Size** (5×5 to 10×10)  
✅ **Click-to-Select Starting Position**  
✅ **Real-time Path Visualization** with move numbers and arrows  
✅ **Solution Database** for tracking and analyzing results  
✅ **Progress Monitoring** with generation/step tracking  
✅ **Performance Statistics** including execution time and success rate

---

## 📁 Project Structure

```
KnightTourProject/
│
├── README.md
├── requirements.txt
├── main.py                    # Main entry point
│
├── config/
│   └── config.py              # Configuration parameters
│
├── core/
│   ├── board.py               # Board representation
│   │
│   └── algorithms/
│       ├── backtracking/
│       │   ├── brute_force.py
│       │   ├── randomized_heuristic.py
│       │   └── warnsdorff.py
│       │
│       ├── cultural/
│       │   ├── cultural_algorithm.py
│       │   ├── individual.py
│       │   ├── belief_space.py
│       │   ├── fitness.py
│       │   └── population.py
│       │
│       └── genetic/          # Additional GA implementations
│           ├── ga_classic.py
│           ├── GA_classic_opt.py
│           └── GA_Warnsdorff.py
│
├── gui/
│   └── visualization.py       # GUI implementation
│
├── db/
│   ├── database.py            # SQLite database manager
│   └── knight_tour_new.db     # Solutions database
│
├── utils/
│   └── utils.py               # Utility functions
│
└── tests/
    ├── test.py                # Extended algorithm testing
    └── test2.py               # Comprehensive comparison tests
```

---

## 🛠️ Installation

### Prerequisites

- **Python 3.7 or higher**
- **tkinter** (usually comes with Python)

### Setup

1. **Clone the repository:**

```bash
git clone <https://github.com/Salmaazoz22/Knights-Tour-Problem-Solver>
cd KnightTourProject
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the application:**

```bash
python main.py
```

---

## 📖 Usage

### GUI Mode

1. **Launch the application:**
   ```bash
   python main.py
   ```

2. **Select board size** (5-10) using the spinner control

3. **Click on any square** to set the knight's starting position

4. **Choose algorithm:**
   - **Cultural Algorithm**: Fast evolutionary approach, typically finds solutions in seconds
   - **Backtracking**: Systematic search with Warnsdorff's heuristic

5. **Click START ♞** to begin solving

6. **Watch** the knight's path being visualized in real-time with:
   - Move numbers on each square
   - Connecting arrows showing the path
   - Knight piece position

7. **Results** are automatically saved to the database with:
   - Complete solution path
   - Execution time
   - Number of generations (for Cultural Algorithm)
   - Fitness score

### Command Line Mode

You can also use the core algorithms directly in your own scripts:

```python
from core.cultural_algorithm import CulturalAlgorithm
from core.board import Board

# Create and run Cultural Algorithm
ca = CulturalAlgorithm(board_size=8, start_pos=0, pop_size=400, max_gen=3000)
solution = ca.run()

if solution:
    print(f"Solution found! Fitness: {solution.fitness}")
    print(f"Path: {solution.chromosome}")
```

---

## 🧬 Algorithms Explained

### Cultural Algorithm

The Cultural Algorithm is an evolutionary computation technique that mimics dual inheritance found in human societies:

**Key Components:**

1. **Population Space**: 
   - Evolving candidate solutions (individuals)
   - Smart initialization using Warnsdorff's heuristic
   - Tournament selection for parent selection
   - Order crossover (OX1) for generating offspring
   - Intelligent mutation operators

2. **Belief Space**:
   - **Situational Knowledge**: Stores best solutions found
   - **Normative Knowledge**: Learns statistical patterns of good moves
   - **Historical Knowledge**: Tracks common successful move sequences

3. **Evolution Process**:
   - Elitism preserves top 10% of population
   - Adaptive mutation based on belief space
   - Stagnation detection (stops after 2000 generations without improvement)
   - Generation tracking and callback support

**Performance:**
- Board 5×5: ~0.001s average
- Board 7×7: ~0.03s average  
- Board 10×10: ~0.08s average
- **Success Rate: 100%** for all tested board sizes

### Backtracking with Warnsdorff's Heuristic

Classic recursive approach optimized with intelligent move selection:

**Key Features:**

1. **Warnsdorff's Rule**: Always choose the square with the fewest onward moves
2. **Randomized Tie-Breaking**: When multiple moves have equal degrees, pick randomly
3. **Efficient Pruning**: Abandons unfruitful paths early
4. **Guaranteed Solution**: Will find a solution if one exists

**Performance:**
- Board 5×5: ~0.0001s average
- Board 7×7: ~0.06s average
- Board 10×10: ~0.001s average
- **Success Rate: 100%** for all tested configurations

---

## ⚙️ Configuration

Edit `config/config.py` to customize algorithm parameters:

```python
# Cultural Algorithm Parameters
CA_POPULATION_SIZE = 400
CA_MAX_GENERATIONS = 3000
CA_CROSSOVER_PROB = 0.9
CA_MUTATION_PROB = 0.3
CA_ELITE_RATIO = 0.1

# Belief Space Parameters
BELIEF_ACCEPTANCE_RATE = 0.25
BELIEF_SITUATIONAL_SIZE = 8
BELIEF_SITUATIONAL_INFLUENCE_PROB = 0.8

# Stagnation Control
STAGNATION_THRESHOLD = 200

# GUI Settings
GUI_CELL_SIZE = 60
GUI_UPDATE_INTERVAL = 5
GUI_ANIMATION_DELAY = 0.2

# Colors
COLOR_LIGHT_SQUARE = "#F0D9B5"
COLOR_DARK_SQUARE = "#B58863"
COLOR_PATH = "#00FF96"
COLOR_KNIGHT = "black"
```

---

## 📊 Database

Solutions are automatically saved to `db/knight_tour_new.db` with SQLite, storing:

- Board size and starting position
- Algorithm used
- Complete solution path (chromosome)
- Fitness score
- Generation count (for Cultural Algorithm)
- Execution time
- Timestamp

**Viewing Saved Solutions:**

```python
from db.database import KnightTourDatabase

db = KnightTourDatabase()

# Get solutions for 8x8 board
solutions = db.get_solutions_by_board_size(8)
for sol in solutions:
    print(f"Solution {sol['id']}: Fitness {sol['fitness']}, "
          f"Time {sol['time_seconds']:.2f}s, "
          f"Generations {sol['generations']}")

# Get best solutions
best = db.get_best_solutions(board_size=8, algorithm="Cultural Algorithm", limit=5)
```

---

## 📈 Performance Comparison

Based on extensive testing across multiple board sizes and starting positions:

### Cultural Algorithm
| Board Size | Avg Time | Success Rate |
|------------|----------|--------------|
| 5×5        | 0.001s   | 100%         |
| 6×6        | 0.0004s  | 100%         |
| 7×7        | 0.03s    | 100%         |
| 8×8        | 0.05s    | 100%         |
| 9×9        | 0.08s    | 100%         |
| 10×10      | 0.08s    | 100%         |

### Randomized Warnsdorff Backtracking
| Board Size | Avg Time | Success Rate |
|------------|----------|--------------|
| 5×5        | 0.0001s  | 100%         |
| 6×6        | 0.0001s  | 100%         |
| 7×7        | 0.06s    | 100%         |
| 8×8        | 0.0003s  | 100%         |
| 9×9        | 0.0005s  | 100%         |
| 10×10      | 0.001s   | 100%         |

---

## 🎓 Educational Value

This project demonstrates key AI concepts:

- **Evolutionary Algorithms**: Population-based optimization
- **Cultural Evolution**: Knowledge accumulation and belief systems
- **Heuristic Search**: Warnsdorff's rule and intelligent pruning
- **Backtracking**: Systematic exhaustive search
- **GUI Programming**: Interactive visualization with Tkinter
- **Software Design**: Modular, extensible architecture
- **Data Persistence**: SQLite database integration
- **Threading**: Non-blocking GUI with background computation

---

## 🐛 Troubleshooting

**Issue**: GUI doesn't start
- **Solution**: Ensure tkinter is installed:
  ```bash
  # On Ubuntu/Debian
  sudo apt-get install python3-tk
  
  # On macOS
  brew install python-tk
  ```

**Issue**: Algorithm is too slow
- **Solution**: Reduce `CA_POPULATION_SIZE` or `CA_MAX_GENERATIONS` in `config/config.py`

**Issue**: Database errors
- **Solution**: Delete `db/knight_tour_new.db` to recreate fresh database

**Issue**: Generation count shows 0
- **Solution**: Ensure you're using the latest version with fixed `CulturalThread`

---

## 📝 Testing

Run comprehensive tests to compare all algorithms:

```bash
# Extended testing for Cultural Algorithm and Backtracking
python test.py

# Complete comparison including all GA variants
python test2.py
```

Results are saved to CSV files:
- `comparison_results_extended.csv`
- `all_solvers_comparison.csv`

---

## 🎯 Key Achievements

✅ **100% Success Rate** on all board sizes (5×5 to 10×10)  
✅ **Sub-second Performance** for most configurations  
✅ **Robust Implementation** with error handling and validation  
✅ **User-Friendly Interface** with real-time visualization  
✅ **Comprehensive Testing** across multiple scenarios  
✅ **Database Integration** for solution tracking and analysis

---

## 📚 References

- **Warnsdorff's Heuristic**: Classic knight's tour algorithm (1823)
- **Cultural Algorithm Framework**: Reynolds, R.G. (1994)
- **Evolutionary Computation**: Principles and applications
- **Chess Programming**: Board representation and move generation

---

## 🎉 Acknowledgments

- **Dr.AmrGhoneim** - Course Instructor, AI310 & CS361
- **Helwan University** - Faculty of Computing & Artificial Intelligence
- **Course TAs** - For guidance and support throughout the project
- **Open Source Community** - For tools and inspiration

---

## 📄 License

This project is created for educational purposes as part of the AI course at Helwan University.

---

## 📧 Contact

For questions or feedback about this project:

- **Course**: AI310 & CS361 Artificial Intelligence
- **Institution**: Helwan University, Faculty of Computing & Artificial Intelligence
- **Department**: Computer Science Department
- **Programmes**: Mainstream, Software Engineering, and Medical Informatics

---

**Made with ♞ and 🧠 by Team [Your Team Name]**

*Fall Semester 1, 2025-2026*
