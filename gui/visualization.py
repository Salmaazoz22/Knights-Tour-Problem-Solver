import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from config.config import *
from core.board import Board
from utils.utils import square_to_position, position_to_square
from db.database import KnightTourDatabase

# Import algorithms
from core.algorithms import CulturalAlgorithm, RandomizedHeuristicBacktracking


class ChessBoard(tk.Canvas):
    """Visual chess board widget with knight animation."""
    
    def __init__(self, parent, n=DEFAULT_BOARD_SIZE, cell_size=GUI_CELL_SIZE):
        super().__init__(parent, width=n*cell_size, height=n*cell_size, bg="white")
        self.n = n
        self.cell_size = cell_size
        self.path = []
        self.current_x = 0
        self.current_y = 0
        self.bind("<Button-1>", self.set_start)
        self.draw_board()

    def set_start(self, event):
        """Allow changing start point before starting solver."""
        j = event.x // self.cell_size
        i = event.y // self.cell_size
        if 0 <= i < self.n and 0 <= j < self.n:
            self.current_x, self.current_y = i, j
            self.path = []
            self.draw_board()

    def update_path(self, chromosome):
        """Update the path from chromosome (list of square indices)."""
        self.path = []
        for sq in chromosome:
            row, col = square_to_position(sq, self.n)
            self.path.append((row, col))
        self.draw_board()

    def draw_board(self):
        """Draw the chessboard with current path."""
        self.delete("all")
        
        # Draw squares
        for i in range(self.n):
            for j in range(self.n):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                color = COLOR_LIGHT_SQUARE if (i+j) % 2 == 0 else COLOR_DARK_SQUARE
                self.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
                
                # Draw move numbers
                if self.path:
                    for idx, (pi, pj) in enumerate(self.path):
                        if pi == i and pj == j:
                            self.create_text(x0+self.cell_size//2, y0+self.cell_size//2,
                                           text=str(idx+1), font=("Arial", 10, "bold"),
                                           fill="red")
                            break
        
        # Draw path lines
        if len(self.path) > 1:
            for idx in range(len(self.path) - 1):
                i1, j1 = self.path[idx]
                i2, j2 = self.path[idx + 1]
                x1 = j1 * self.cell_size + self.cell_size // 2
                y1 = i1 * self.cell_size + self.cell_size // 2
                x2 = j2 * self.cell_size + self.cell_size // 2
                y2 = i2 * self.cell_size + self.cell_size // 2
                self.create_line(x1, y1, x2, y2, fill=COLOR_PATH, width=2, arrow=tk.LAST)
        
        # Draw knight at starting position
        if self.path:
            i, j = self.path[0]
        else:
            i, j = self.current_x, self.current_y
        x0 = j * self.cell_size + self.cell_size // 2
        y0 = i * self.cell_size + self.cell_size // 2
        self.create_text(x0, y0, text="♞", font=("Segoe UI Symbol", 24), fill=COLOR_KNIGHT)


class CulturalThread(threading.Thread):
    """Thread to run Cultural Algorithm without freezing GUI."""
    
    def __init__(self, board_widget, start_x, start_y, board_size, callback=None):
        super().__init__()
        self.board_widget = board_widget
        self.start_x = start_x
        self.start_y = start_y
        self.board_size = board_size
        self.callback = callback
        self.running = True
        self.ca = None
        self.time_seconds = None
        self.generations = None

    def run(self):
        import time as time_module
        
        start_pos = self.start_x * self.board_size + self.start_y
        
        print(f"🚀 Cultural Algorithm started - Board: {self.board_size}x{self.board_size}, Start: {start_pos}")
        
        start_time = time_module.time()
        
        # Track the generation number
        self.generations = 0
        
        def update_callback(gen, best_individual, target):
            if not self.running:
                return
            
            # Update generation counter
            self.generations = gen
            
            if gen % GUI_UPDATE_INTERVAL == 0:
                self.board_widget.update_path(best_individual.chromosome)
                self.board_widget.update()
        
        self.ca = CulturalAlgorithm(
            board_size=self.board_size,
            start_pos=start_pos,
            pop_size=CA_POPULATION_SIZE,
            max_gen=CA_MAX_GENERATIONS,
            callback=update_callback
        )
        
        result = self.ca.run()
        
        self.time_seconds = time_module.time() - start_time
        
        if result:
            self.board_widget.update_path(result.chromosome)
            is_complete = result.fitness == self.board_size * self.board_size - 1
            print(f"✅ Cultural Algorithm complete: {is_complete}, Fitness: {result.fitness}, Time: {self.time_seconds:.2f}s, Generations: {self.generations}")
            if self.callback:
                self.callback(is_complete, result.chromosome if result else None)
        else:
            print(f"❌ Cultural Algorithm failed")
            if self.callback:
                self.callback(False, None)

    def stop(self):
        self.running = False
        if self.ca:
            self.ca.stop()

class BacktrackingThread(threading.Thread):
    """Thread for Backtracking algorithm."""
    
    def __init__(self, board_widget, start_x, start_y, board_size, callback=None):
        super().__init__()
        self.board_widget = board_widget
        self.start_x = start_x
        self.start_y = start_y
        self.board_size = board_size
        self.callback = callback
        self.running = True
        self.solver = None
        self.time_seconds = None

    def run(self):
        import time as time_module
        
        board = Board(self.board_size)
        start_square = self.start_x * self.board_size + self.start_y
        
        print(f"🚀 Backtracking started - Board: {self.board_size}x{self.board_size}, Start: {start_square}")
        
        start_time = time_module.time()
        
        def viz_callback(path, step):
            if not self.running:
                return
            self.board_widget.update_path(path)
            self.board_widget.update()
            time.sleep(0.1)
        
        self.solver = RandomizedHeuristicBacktracking(board)
        result = self.solver.solve(start_square, callback=viz_callback)
        
        self.time_seconds = time_module.time() - start_time
        
        print(f"🔍 Backtracking result: {result}")
        print(f"🔍 Result type: {type(result)}")
        print(f"🔍 Result length: {len(result) if result else 0}")
        print(f"🔍 Expected length: {self.board_size * self.board_size}")
        print(f"⏱️ Time taken: {self.time_seconds:.2f}s")
        
        if result:
            self.board_widget.update_path(result)
            is_complete = len(result) == self.board_size * self.board_size
            print(f"🔍 Is complete: {is_complete}")
            print(f"🔍 Callback exists: {self.callback is not None}")
            
            if self.callback:
                print(f"✅ Calling callback with success={is_complete}, chromosome length={len(result)}")
                self.callback(is_complete, result)
        else:
            print(f"❌ No result found")
            if self.callback:
                self.callback(False, None)

    def stop(self):
        self.running = False
        if self.solver:
            self.solver.stop()


class KnightTourGUI:
    """Main GUI application."""
    
    def __init__(self, root):
        self.root = root
        root.title("Knight's Tour Problem Solver ♞")
        root.resizable(False, False)
        
        # Database
        self.db = KnightTourDatabase()
        print(f"📁 Database path: {self.db.db_path}")
        
        # Main container
        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left: Board
        board_frame = tk.Frame(main_frame)
        board_frame.pack(side="left", padx=10)
        
        self.board_widget = ChessBoard(board_frame, n=DEFAULT_BOARD_SIZE)
        self.board_widget.pack()
        
        # Right: Controls
        ctrl_frame = tk.Frame(main_frame)
        ctrl_frame.pack(side="right", fill="y", padx=10)
        
        # Title
        tk.Label(ctrl_frame, text="Knight's Tour Solver", 
                font=("Arial", 16, "bold")).pack(pady=10)
        
        # Board size
        tk.Label(ctrl_frame, text="Board Size:", font=("Arial", 11)).pack(pady=5)
        self.size_var = tk.IntVar(value=DEFAULT_BOARD_SIZE)
        size_frame = tk.Frame(ctrl_frame)
        size_frame.pack()
        tk.Spinbox(size_frame, from_=MIN_BOARD_SIZE, to=MAX_BOARD_SIZE, 
                  textvariable=self.size_var, width=8, font=("Arial", 11),
                  command=self.resize_board).pack()
        
        # Algorithm selection
        tk.Label(ctrl_frame, text="Algorithm:", font=("Arial", 11)).pack(pady=(15, 5))
        self.algo_var = tk.StringVar(value="Cultural Algorithm")
        ttk.Combobox(ctrl_frame, textvariable=self.algo_var,
                    values=["Cultural Algorithm", "Backtracking"],
                    state="readonly", width=20).pack()
        
        # Instructions
        tk.Label(ctrl_frame, text="\nClick a square\nto set starting position", 
                font=("Arial", 9), fg="gray").pack(pady=5)
        
        # Start button
        self.start_btn = tk.Button(ctrl_frame, text="START ♞", 
                                   bg=COLOR_START_BUTTON, fg="white",
                                   font=("Arial", 14, "bold"), width=15,
                                   command=self.start)
        self.start_btn.pack(pady=15)
        
        # Stop button
        self.stop_btn = tk.Button(ctrl_frame, text="STOP", 
                                  bg=COLOR_STOP_BUTTON, fg="white",
                                  font=("Arial", 12), width=15,
                                  state="disabled", command=self.stop)
        self.stop_btn.pack(pady=5)
        
        # Status
        self.status_label = tk.Label(ctrl_frame, text="Ready", 
                                     font=("Arial", 10), fg="green")
        self.status_label.pack(pady=10)
        
        # Progress
        self.progress_label = tk.Label(ctrl_frame, text="", 
                                       font=("Arial", 9))
        self.progress_label.pack()
        
        self.solver_thread = None

    def resize_board(self):
        """Resize the board when size changes."""
        n = self.size_var.get()
        self.board_widget.n = n
        self.board_widget.path = []
        self.board_widget.current_x = 0
        self.board_widget.current_y = 0
        self.board_widget.config(width=n*self.board_widget.cell_size, 
                                height=n*self.board_widget.cell_size)
        self.board_widget.draw_board()

    def start(self):
        """Start the selected algorithm."""
        if self.solver_thread and self.solver_thread.is_alive():
            messagebox.showwarning("Warning", "Algorithm is already running!")
            return
        
        self.status_label.config(text="Running...", fg="orange")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress_label.config(text="")
        
        algo = self.algo_var.get()
        
        if algo == "Backtracking":
            self.solver_thread = BacktrackingThread(
                self.board_widget,
                self.board_widget.current_x,
                self.board_widget.current_y,
                self.board_widget.n,
                callback=self.on_complete
            )
        else:  # Cultural Algorithm
            self.solver_thread = CulturalThread(
                self.board_widget,
                self.board_widget.current_x,
                self.board_widget.current_y,
                self.board_widget.n,
                callback=self.on_complete
            )
        
        self.solver_thread.start()
        self.monitor_thread()

    def stop(self):
        """Stop the running algorithm."""
        if self.solver_thread:
            self.solver_thread.stop()
            self.status_label.config(text="Stopped", fg="red")
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")

    def on_complete(self, success, chromosome):
        """Callback when algorithm completes."""
        print(f"🎯 on_complete called - Success: {success}, Chromosome: {chromosome[:5] if chromosome else None}...")
        
        if success and chromosome:
            messagebox.showinfo("Success! 🎉", "Complete Knight's Tour found!")
            # Save to database
            start_sq = position_to_square(self.board_widget.current_x, 
                                        self.board_widget.current_y, 
                                        self.board_widget.n)
            try:
                # Get time and generations from thread
                time_seconds = None
                generations = None
                
                if hasattr(self.solver_thread, 'time_seconds'):
                    time_seconds = self.solver_thread.time_seconds
                
                if hasattr(self.solver_thread, 'generations'):
                    generations = self.solver_thread.generations
                
                print(f"💾 Saving to database - Algorithm: {self.algo_var.get()}, Start: {start_sq}, Length: {len(chromosome)}, Time: {time_seconds}, Generations: {generations}")
                
                solution_id = self.db.save_solution(
                    board_size=self.board_widget.n,
                    start_pos=start_sq,
                    algorithm=self.algo_var.get(),
                    chromosome=chromosome,
                    fitness=len(chromosome) - 1,
                    generations=generations,
                    time_seconds=time_seconds
                )
                print(f"✅ Solution saved with ID: {solution_id}")
            except Exception as e:
                print(f"❌ Error saving to database: {e}")
                import traceback
                traceback.print_exc()
                messagebox.showerror("Database Error", f"Failed to save solution: {e}")
        else:
            messagebox.showinfo("Result", "No complete tour found.")

    def monitor_thread(self):
        """Monitor if thread is still running."""
        if self.solver_thread and self.solver_thread.is_alive():
            self.root.after(100, self.monitor_thread)
        else:
            self.status_label.config(text="Finished", fg="green")
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")