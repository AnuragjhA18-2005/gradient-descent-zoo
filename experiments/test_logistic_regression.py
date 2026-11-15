import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tasks.logistic_regression import generate_synthetic_data, LogisticRegression


class TestLogisticRegressionTask(unittest.TestCase):
    def test_synthetic_data_generation(self):
        X, y = generate_synthetic_data(n_samples=200, random_state=42)
        self.assertEqual(X.shape, (200, 2))
        self.assertEqual(y.shape, (200, 1))
        unique_labels = np.unique(y)
        self.assertTrue(np.array_equal(np.sort(unique_labels), np.array([0.0, 1.0])))

    def test_forward_pass_bounds(self):
        model = LogisticRegression(in_features=2, seed=42)
        X = np.random.randn(50, 2)
        y_pred = model.forward(X)
        self.assertEqual(y_pred.shape, (50, 1))
        self.assertTrue(np.all(y_pred >= 0.0) and np.all(y_pred <= 1.0))

    def test_loss_computation(self):
        y_true = np.array([[1.0], [0.0], [1.0]])
        y_pred = np.array([[0.9], [0.1], [0.8]])
        loss = LogisticRegression.compute_loss(y_true, y_pred)
        self.assertGreater(loss, 0.0)
        self.assertLess(loss, 0.5)

    def test_gradient_numerical_check(self):
        """Verify analytical backward pass against numerical finite-difference gradients."""
        rng = np.random.RandomState(42)
        X = rng.randn(25, 2)
        y = (rng.rand(25, 1) > 0.5).astype(np.float64)

        model = LogisticRegression(in_features=2, seed=42)
        y_pred = model.forward(X)
        ana_grad_W, ana_grad_b = model.backward(X, y, y_pred)

        # Finite difference for W
        h = 1e-6
        num_grad_W = np.zeros_like(model.W)
        for i in range(model.W.shape[0]):
            for j in range(model.W.shape[1]):
                model.W[i, j] += h
                loss_plus = model.compute_loss(y, model.forward(X))
                model.W[i, j] -= 2 * h
                loss_minus = model.compute_loss(y, model.forward(X))
                model.W[i, j] += h  # restore
                num_grad_W[i, j] = (loss_plus - loss_minus) / (2 * h)

        # Finite difference for b
        model.b[0] += h
        loss_plus = model.compute_loss(y, model.forward(X))
        model.b[0] -= 2 * h
        loss_minus = model.compute_loss(y, model.forward(X))
        model.b[0] += h  # restore
        num_grad_b = np.array([(loss_plus - loss_minus) / (2 * h)])

        np.testing.assert_allclose(ana_grad_W, num_grad_W, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ana_grad_b, num_grad_b, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
