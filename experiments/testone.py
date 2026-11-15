

from optimizers import Optimizer


if __name__ == "__main__":
    try:
        op = Optimizer(-0.1)
        op.step([1, 2, 3], [0.1, 0.2, 0.3])
    except ValueError as e:
        print(f"Caught expected error: {e}")

    