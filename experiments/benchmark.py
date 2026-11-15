import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tasks.logistic_regression import run_benchmark

if __name__ == "__main__":
    run_benchmark()
