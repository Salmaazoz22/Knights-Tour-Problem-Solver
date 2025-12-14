#!/usr/bin/env python3
"""
Knight's Tour Problem Solver
============================
Main entry point for the application.

This file launches the GUI application that allows users to solve
the Knight's Tour problem using either Cultural Algorithm or Backtracking.

Usage:
    python main.py

Author: [Salma]
Date: 2025
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from gui.visualization import KnightTourGUI


def main():
    """
    Main function to start the Knight's Tour application.
    
    Creates the main Tkinter window and initializes the GUI.
    The application will run until the user closes the window.
    """
    # Create the main window
    root = tk.Tk()
    
    # Initialize the GUI application
    app = KnightTourGUI(root)
    
    # Start the Tkinter event loop
    # This keeps the window open and responsive to user actions
    root.mainloop()


if __name__ == "__main__":
    main()