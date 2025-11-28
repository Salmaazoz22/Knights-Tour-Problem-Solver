# knight_tour_gui_simple_fixed.py
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import random

# --------------------- BOARD & MOVES ---------------------
def get_knight_moves(x, y, n):
    moves = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
    return [(x+dx, y+dy) for dx,dy in moves if 0<=x+dx<n and 0<=y+dy<n]

class Board:
    def __init__(self, n):
        self.n = n
        self.grid = [[0]*n for _ in range(n)]
    def place(self, x, y, val): self.grid[x][y] = val
    def remove(self, x, y): self.grid[x][y] = 0
    def is_empty(self, x, y): return 0<=x<self.n and 0<=y<self.n and self.grid[x][y]==0

# --------------------- SOLVERS ---------------------
class BacktrackingSolver(threading.Thread):
    def __init__(self, board_widget, start_x, start_y):
        super().__init__()
        self.board_widget = board_widget
        self.start_x = start_x
        self.start_y = start_y
        self.running = True

    def run(self):
        self.board_widget.board = Board(self.board_widget.n)
        self.board_widget.path = [(self.start_x,self.start_y)]
        self.board_widget.board.place(self.start_x,self.start_y,1)
        self.board_widget.draw_board()
        time.sleep(0.5)
        if not self.solve(self.start_x,self.start_y,2):
            messagebox.showinfo("Result","No full tour found!")
        else:
            messagebox.showinfo("Result","TOUR COMPLETED SUCCESSFULLY! 🎉")

    def solve(self, x, y, step):
        if not self.running: return False
        if step>self.board_widget.n**2: return True
        candidates = get_knight_moves(x,y,self.board_widget.n)
        candidates.sort(key=lambda p: len([m for m in get_knight_moves(*p,self.board_widget.n)
                                           if self.board_widget.board.is_empty(*m)]))
        for nx, ny in candidates:
            if self.board_widget.board.is_empty(nx, ny):
                self.board_widget.path.append((nx,ny))
                self.board_widget.board.place(nx,ny,step)
                self.board_widget.draw_board()
                time.sleep(0.2)
                if self.solve(nx, ny, step+1):
                    return True
                # backtrack
                self.board_widget.path.pop()
                self.board_widget.board.remove(nx,ny)
                self.board_widget.draw_board()
                time.sleep(0.2)
        return False

    def stop(self): self.running=False

class CulturalSolver(threading.Thread):
    def __init__(self, board_widget, start_x, start_y):
        super().__init__()
        self.board_widget = board_widget
        self.start_x = start_x
        self.start_y = start_y
        self.running = True

    def run(self):
        n = self.board_widget.n
        target = n*n
        pop_size = 40

        def random_tour():
            tour = [(self.start_x,self.start_y)]
            visited = {(self.start_x,self.start_y)}
            x,y = self.start_x,self.start_y
            while len(tour)<target:
                opts = [p for p in get_knight_moves(x,y,n) if p not in visited]
                if not opts: break
                nx,ny=random.choice(opts)
                tour.append((nx,ny))
                visited.add((nx,ny))
                x,y = nx,ny
            return tour

        population = [random_tour() for _ in range(pop_size)]
        belief = {}

        for gen in range(1,5000):
            if not self.running: break
            scored = [(tour,len(set(tour))) for tour in population]
            scored.sort(key=lambda x:x[1],reverse=True)
            best_tour,best_fit = scored[0]
            self.board_widget.path = best_tour[:best_fit]
            self.board_widget.draw_board()
            time.sleep(0.05)
            if best_fit==target:
                messagebox.showinfo("Result","TOUR COMPLETED SUCCESSFULLY! 🎉")
                return
            elites = [t[0] for t in scored[:6]]
            new_pop = elites[:]
            while len(new_pop)<pop_size:
                p = random.choice(elites)
                child = p[:random.randint(2,len(p)//2)]
                while len(child)<target and child:
                    curr = child[-1]
                    opts = belief.get(curr,get_knight_moves(*curr,n))
                    opts = [p for p in opts if p not in child]
                    if not opts: break
                    child.append(random.choice(opts))
                new_pop.append(child)
            population=new_pop
            for i in range(len(best_tour)-1):
                belief.setdefault(best_tour[i],[]).append(best_tour[i+1])
        messagebox.showinfo("Result","No full tour found")

    def stop(self): self.running=False

# --------------------- BOARD WIDGET ---------------------
class ChessBoard(tk.Canvas):
    def __init__(self, parent, n=8, cell_size=60):
        super().__init__(parent,width=n*cell_size,height=n*cell_size,bg="white")
        self.n = n
        self.cell_size = cell_size
        self.board = Board(n)
        self.path = []
        self.current_x = 0
        self.current_y = 0
        self.bind("<Button-1>", self.set_start)
        self.draw_board()

    def set_start(self, event):
        """Allow changing start point anytime before starting the solver"""
        j = event.x // self.cell_size
        i = event.y // self.cell_size
        if 0 <= i < self.n and 0 <= j < self.n:
            self.current_x, self.current_y = i, j
            # reset board and path
            self.board = Board(self.n)
            self.path = [(i, j)]
            self.board.place(i, j, 1)
            self.draw_board()

    def draw_board(self):
        self.delete("all")
        colors=["#F0D9B5","#B58863"]
        for i in range(self.n):
            for j in range(self.n):
                x0=j*self.cell_size
                y0=i*self.cell_size
                x1=x0+self.cell_size
                y1=y0+self.cell_size
                self.create_rectangle(x0,y0,x1,y1,fill=colors[(i+j)%2])
                if self.board.grid[i][j]>0:
                    self.create_text(x0+self.cell_size//2,y0+self.cell_size//2,
                                     text=str(self.board.grid[i][j]),font=("Arial",14,"bold"))
        # path
        for idx,(i,j) in enumerate(self.path):
            if (i,j)==(self.current_x,self.current_y): continue
            x0=j*self.cell_size+self.cell_size//4
            y0=i*self.cell_size+self.cell_size//4
            self.create_oval(x0,y0,x0+self.cell_size//2,y0+self.cell_size//2,
                             fill="#00FF96")
        # knight
        x0=self.current_y*self.cell_size+self.cell_size//2
        y0=self.current_x*self.cell_size+self.cell_size//2
        self.create_text(x0,y0,text="♞",font=("Segoe UI Symbol",24))

    def update_board(self):
        self.draw_board()
        self.update()

# --------------------- MAIN APP ---------------------
class KnightTourApp:
    def __init__(self, root):
        self.root=root
        root.title("Knight's Tour Problem Solver ♞")
        self.board_widget=ChessBoard(root)
        self.board_widget.pack(side="left",padx=20,pady=20)

        ctrl = tk.Frame(root)
        ctrl.pack(side="right",fill="y",padx=10,pady=10)
        tk.Label(ctrl,text="Board Size:").pack()
        self.size_var=tk.IntVar(value=8)
        tk.Spinbox(ctrl,from_=5,to=10,textvariable=self.size_var,width=5,command=self.resize_board).pack()

        tk.Label(ctrl,text="Algorithm:").pack(pady=(10,0))
        self.algo_var=tk.StringVar(value="Backtracking")
        ttk.Combobox(ctrl,textvariable=self.algo_var,values=["Backtracking","Cultural"],state="readonly").pack()

        self.start_btn=tk.Button(ctrl,text="START ♞",bg="#d40000",fg="white",font=("Arial",16,"bold"),command=self.start)
        self.start_btn.pack(pady=20)
        self.solver_thread=None

    def resize_board(self):
        n=self.size_var.get()
        self.board_widget.n=n
        self.board_widget.board=Board(n)
        self.board_widget.path=[]
        self.board_widget.current_x=0
        self.board_widget.current_y=0
        self.board_widget.config(width=n*self.board_widget.cell_size,height=n*self.board_widget.cell_size)
        self.board_widget.draw_board()

    def start(self):
        if not self.board_widget.path:
            messagebox.showinfo("Info","Click a square to set start position!")
            return
        if self.solver_thread and self.solver_thread.is_alive(): return
        algo=self.algo_var.get()
        if algo=="Backtracking":
            self.solver_thread=BacktrackingSolver(self.board_widget,
                                                  self.board_widget.current_x,
                                                  self.board_widget.current_y)
        else:
            self.solver_thread=CulturalSolver(self.board_widget,
                                              self.board_widget.current_x,
                                              self.board_widget.current_y)
        self.solver_thread.start()

if __name__=="__main__":
    root=tk.Tk()
    app=KnightTourApp(root)
    root.mainloop()
