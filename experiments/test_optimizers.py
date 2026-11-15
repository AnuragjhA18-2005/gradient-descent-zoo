import os
import sys
import unittest
import numpy as np

# Add project root to sys.path so optimizers package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optimizers import Optimizer, SGD, Momentum, NAG, AdaGrad, AdaDelta, RMSprop, Adam


class TestOptimizerBase(unittest.TestCase):
    def test_invalid_lr(self):
        with self.assertRaises(ValueError):
            Optimizer(-0.1)
        with self.assertRaises(ValueError):
            Optimizer(0.0)

    def test_valid_lr(self):
        opt = Optimizer(0.05)
        self.assertEqual(opt.lr, 0.05)
        self.assertIsInstance(opt.lr, float)

    def test_not_implemented_step(self):
        opt = Optimizer(0.1)
        with self.assertRaises(NotImplementedError):
            opt.step([np.array([1.0])], [np.array([0.1])])

    def test_check_length(self):
        opt = Optimizer(0.1)
        # Matching lengths
        opt.check_length([np.array([1.0])], [np.array([0.1])])
        # Mismatched lengths
        with self.assertRaises(ValueError):
            opt.check_length([np.array([1.0])], [np.array([0.1]), np.array([0.2])])


class TestCommonOptimizerContracts(unittest.TestCase):
    def setUp(self):
        self.optimizers = [
            ("SGD", SGD(lr=0.1)),
            ("Momentum", Momentum(lr=0.1, momentum=0.9)),
            ("NAG", NAG(lr=0.1, momentum=0.9)),
            ("AdaGrad", AdaGrad(lr=0.1, epsilon=1e-8)),
            ("AdaDelta", AdaDelta(rho=0.9, epsilon=1e-8)),
            ("RMSprop", RMSprop(lr=0.01, rho=0.9, epsilon=1e-8)),
            ("Adam", Adam(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)),
        ]

    def test_in_place_update(self):
        """Verify that parameters are mutated strictly in place (same array buffer)."""
        for name, opt in self.optimizers:
            with self.subTest(optimizer=name):
                p = np.array([1.0, 2.0, 3.0], dtype=np.float64)
                original_id = id(p)
                original_data_ptr = p.__array_interface__["data"][0]
                g = np.array([0.1, 0.2, 0.3], dtype=np.float64)

                params = [p]
                grads = [g]
                opt.step(params, grads)

                # Parameter values should have changed
                self.assertFalse(np.allclose(p, [1.0, 2.0, 3.0]))
                # Array identity and data buffer pointer must remain identical
                self.assertEqual(id(p), original_id)
                self.assertEqual(p.__array_interface__["data"][0], original_data_ptr)
                self.assertIs(params[0], p)

    def test_length_mismatch_raises(self):
        """Passing mismatched params and grads lists must raise ValueError."""
        for name, opt in self.optimizers:
            with self.subTest(optimizer=name):
                p = [np.array([1.0]), np.array([2.0])]
                g = [np.array([0.1])]
                with self.assertRaises(ValueError):
                    opt.step(p, g)

    def test_shape_mismatch_raises(self):
        """Passing a gradient with differing shape must raise ValueError."""
        for name, opt in self.optimizers:
            with self.subTest(optimizer=name):
                p = [np.zeros((2, 3))]
                g = [np.zeros((3, 2))]
                with self.assertRaises(ValueError):
                    opt.step(p, g)

    def test_multidimensional_tensors(self):
        """Ensure updates work correctly on multidimensional arrays (1D, 2D, 3D)."""
        for name, opt in self.optimizers:
            with self.subTest(optimizer=name):
                p1 = np.ones((5,))
                p2 = np.ones((3, 4))
                p3 = np.ones((2, 3, 2))
                g1 = np.full((5,), 0.1)
                g2 = np.full((3, 4), 0.2)
                g3 = np.full((2, 3, 2), 0.3)

                opt.step([p1, p2, p3], [g1, g2, g3])
                self.assertTrue(np.all(p1 < 1.0))
                self.assertTrue(np.all(p2 < 1.0))
                self.assertTrue(np.all(p3 < 1.0))


class TestSGD(unittest.TestCase):
    def test_numerical_accuracy(self):
        lr = 0.05
        opt = SGD(lr=lr)
        p = np.array([2.0, -1.0], dtype=np.float64)
        g1 = np.array([0.4, -0.2], dtype=np.float64)

        # Step 1: theta_1 = theta_0 - lr * g1
        opt.step([p], [g1])
        expected_p1 = np.array([2.0, -1.0]) - lr * g1
        np.testing.assert_allclose(p, expected_p1, rtol=1e-7)

        # Step 2: theta_2 = theta_1 - lr * g2
        g2 = np.array([0.1, 0.5], dtype=np.float64)
        opt.step([p], [g2])
        expected_p2 = expected_p1 - lr * g2
        np.testing.assert_allclose(p, expected_p2, rtol=1e-7)


class TestMomentum(unittest.TestCase):
    def test_invalid_momentum(self):
        with self.assertRaises(ValueError):
            Momentum(lr=0.01, momentum=-0.1)
        with self.assertRaises(ValueError):
            Momentum(lr=0.01, momentum=1.0)
        with self.assertRaises(ValueError):
            Momentum(lr=0.01, momentum=1.5)

    def test_numerical_accuracy(self):
        """
        Paper Equation 4:
        v_t = gamma * v_{t-1} + lr * g_t
        theta_t = theta_{t-1} - v_t
        """
        lr = 0.1
        gamma = 0.9
        opt = Momentum(lr=lr, momentum=gamma)

        p = np.array([5.0, -3.0], dtype=np.float64)
        g1 = np.array([1.0, -2.0], dtype=np.float64)

        # Step 1: v_0 = 0 -> v_1 = 0.9 * 0 + 0.1 * g1 = [0.1, -0.2]
        # p_1 = [5.0, -3.0] - [0.1, -0.2] = [4.9, -2.8]
        opt.step([p], [g1])
        expected_v1 = lr * g1
        expected_p1 = np.array([5.0, -3.0]) - expected_v1
        np.testing.assert_allclose(opt.velocity[0], expected_v1, rtol=1e-7)
        np.testing.assert_allclose(p, expected_p1, rtol=1e-7)

        # Step 2: v_2 = 0.9 * v_1 + 0.1 * g2
        g2 = np.array([0.5, 1.0], dtype=np.float64)
        opt.step([p], [g2])
        expected_v2 = gamma * expected_v1 + lr * g2
        expected_p2 = expected_p1 - expected_v2
        np.testing.assert_allclose(opt.velocity[0], expected_v2, rtol=1e-7)
        np.testing.assert_allclose(p, expected_p2, rtol=1e-7)


class TestNAG(unittest.TestCase):
    def test_invalid_momentum(self):
        with self.assertRaises(ValueError):
            NAG(lr=0.01, momentum=-0.1)
        with self.assertRaises(ValueError):
            NAG(lr=0.01, momentum=1.0)

    def test_numerical_accuracy(self):
        """
        Paper Equation 29 (Dozat reformulation):
        v_t = gamma * v_{t-1} + lr * g_t
        theta_t = theta_{t-1} - (gamma * v_t + lr * g_t)
        """
        lr = 0.1
        gamma = 0.9
        opt = NAG(lr=lr, momentum=gamma)

        p = np.array([5.0, -3.0], dtype=np.float64)
        g1 = np.array([1.0, -2.0], dtype=np.float64)

        # Step 1: v_0 = 0 -> v_1 = 0.1 * g1 = [0.1, -0.2]
        # update_1 = gamma * v_1 + lr * g1 = 0.9 * [0.1, -0.2] + 0.1 * [1.0, -2.0] = [0.19, -0.38]
        # p_1 = [5.0, -3.0] - [0.19, -0.38] = [4.81, -2.62]
        opt.step([p], [g1])
        expected_v1 = lr * g1
        expected_update1 = gamma * expected_v1 + lr * g1
        expected_p1 = np.array([5.0, -3.0]) - expected_update1
        np.testing.assert_allclose(opt.velocity[0], expected_v1, rtol=1e-7)
        np.testing.assert_allclose(p, expected_p1, rtol=1e-7)

        # Step 2: v_2 = gamma * v_1 + lr * g2
        g2 = np.array([0.5, 1.0], dtype=np.float64)
        opt.step([p], [g2])
        expected_v2 = gamma * expected_v1 + lr * g2
        expected_update2 = gamma * expected_v2 + lr * g2
        expected_p2 = expected_p1 - expected_update2
        np.testing.assert_allclose(opt.velocity[0], expected_v2, rtol=1e-7)
        np.testing.assert_allclose(p, expected_p2, rtol=1e-7)


class TestAdaGrad(unittest.TestCase):
    def test_invalid_epsilon(self):
        with self.assertRaises(ValueError):
            AdaGrad(lr=0.01, epsilon=0.0)
        with self.assertRaises(ValueError):
            AdaGrad(lr=0.01, epsilon=-1e-5)

    def test_default_values(self):
        opt = AdaGrad()
        self.assertEqual(opt.lr, 0.01)
        self.assertEqual(opt.epsilon, 1e-8)

    def test_numerical_accuracy(self):
        """
        Paper Equations 8 & 9:
        G_t = G_{t-1} + g_t^2
        theta_t = theta_{t-1} - (lr / sqrt(G_t + eps)) * g_t
        """
        lr = 0.05
        eps = 1e-6
        opt = AdaGrad(lr=lr, epsilon=eps)

        p = np.array([3.0, -4.0], dtype=np.float64)
        g1 = np.array([0.6, -0.8], dtype=np.float64)

        # Step 1:
        # G_1 = g1^2 = [0.36, 0.64]
        # p_1 = p_0 - (lr / sqrt(G_1 + eps)) * g1
        opt.step([p], [g1])
        expected_G1 = g1 ** 2
        expected_denom1 = np.sqrt(expected_G1 + eps)
        expected_p1 = np.array([3.0, -4.0]) - (lr / expected_denom1) * g1
        np.testing.assert_allclose(opt.accumulated_grads[0], expected_G1, rtol=1e-7)
        np.testing.assert_allclose(p, expected_p1, rtol=1e-7)

        # Step 2:
        # G_2 = G_1 + g2^2
        g2 = np.array([0.2, 0.3], dtype=np.float64)
        opt.step([p], [g2])
        expected_G2 = expected_G1 + g2 ** 2
        expected_denom2 = np.sqrt(expected_G2 + eps)
        expected_p2 = expected_p1 - (lr / expected_denom2) * g2
        np.testing.assert_allclose(opt.accumulated_grads[0], expected_G2, rtol=1e-7)
        np.testing.assert_allclose(p, expected_p2, rtol=1e-7)


class TestAdaDelta(unittest.TestCase):
    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            AdaDelta(rho=-0.1)
        with self.assertRaises(ValueError):
            AdaDelta(rho=1.0)
        with self.assertRaises(ValueError):
            AdaDelta(epsilon=0.0)
        with self.assertRaises(ValueError):
            AdaDelta(epsilon=-1e-6)

    def test_default_values(self):
        opt = AdaDelta()
        self.assertEqual(opt.lr, 1.0)
        self.assertEqual(opt.rho, 0.9)
        self.assertEqual(opt.epsilon, 1e-8)

    def test_numerical_accuracy(self):
        """
        Paper Section 4.4, Equations 10, 15, 16, 17:
        E[g^2]_t = rho * E[g^2]_{t-1} + (1 - rho) * g_t^2
        RMS[g]_t = sqrt(E[g^2]_t + eps)
        RMS[delta_theta]_{t-1} = sqrt(E[delta_theta^2]_{t-1} + eps)
        delta_theta_t = - (RMS[delta_theta]_{t-1} / RMS[g]_t) * g_t
        E[delta_theta^2]_t = rho * E[delta_theta^2]_{t-1} + (1 - rho) * delta_theta_t^2
        theta_t = theta_{t-1} + delta_theta_t
        """
        rho = 0.95
        eps = 1e-6
        opt = AdaDelta(rho=rho, epsilon=eps)

        p = np.array([3.0, -2.0], dtype=np.float64)
        g1 = np.array([0.4, 0.2], dtype=np.float64)

        # Step 1:
        # E[g^2]_1 = (1 - 0.95) * g1^2 = 0.05 * [0.16, 0.04] = [0.008, 0.002]
        # RMS[g]_1 = sqrt(E[g^2]_1 + eps)
        # RMS[delta_theta]_0 = sqrt(0 + eps) = sqrt(eps)
        # delta_theta_1 = - (RMS[delta_theta]_0 / RMS[g]_1) * g1
        # E[delta_theta^2]_1 = 0.05 * delta_theta_1^2
        # p_1 = p_0 + delta_theta_1
        opt.step([p], [g1])
        expected_eg2_1 = (1.0 - rho) * (g1 ** 2)
        expected_rms_g1 = np.sqrt(expected_eg2_1 + eps)
        expected_rms_u0 = np.sqrt(0.0 + eps)
        expected_delta1 = - (expected_rms_u0 / expected_rms_g1) * g1
        expected_eu2_1 = (1.0 - rho) * (expected_delta1 ** 2)
        expected_p1 = np.array([3.0, -2.0]) + expected_delta1

        np.testing.assert_allclose(opt.avg_squared_grads[0], expected_eg2_1, rtol=1e-7)
        np.testing.assert_allclose(opt.avg_squared_updates[0], expected_eu2_1, rtol=1e-7)
        np.testing.assert_allclose(p, expected_p1, rtol=1e-7)

        # Step 2:
        g2 = np.array([0.1, -0.3], dtype=np.float64)
        opt.step([p], [g2])
        expected_eg2_2 = rho * expected_eg2_1 + (1.0 - rho) * (g2 ** 2)
        expected_rms_g2 = np.sqrt(expected_eg2_2 + eps)
        expected_rms_u1 = np.sqrt(expected_eu2_1 + eps)
        expected_delta2 = - (expected_rms_u1 / expected_rms_g2) * g2
        expected_eu2_2 = rho * expected_eu2_1 + (1.0 - rho) * (expected_delta2 ** 2)
        expected_p2 = expected_p1 + expected_delta2

        np.testing.assert_allclose(opt.avg_squared_grads[0], expected_eg2_2, rtol=1e-7)
        np.testing.assert_allclose(opt.avg_squared_updates[0], expected_eu2_2, rtol=1e-7)
        np.testing.assert_allclose(p, expected_p2, rtol=1e-7)


class TestRMSprop(unittest.TestCase):
    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            RMSprop(rho=-0.1)
        with self.assertRaises(ValueError):
            RMSprop(rho=1.0)
        with self.assertRaises(ValueError):
            RMSprop(epsilon=0.0)
        with self.assertRaises(ValueError):
            RMSprop(epsilon=-1e-6)

    def test_default_values(self):
        opt = RMSprop()
        self.assertEqual(opt.lr, 0.001)
        self.assertEqual(opt.rho, 0.9)
        self.assertEqual(opt.epsilon, 1e-8)

    def test_numerical_accuracy(self):
        """
        Paper Section 4.5, Equation 18:
        E[g^2]_t = rho * E[g^2]_{t-1} + (1 - rho) * g_t^2
        theta_{t+1} = theta_t - (lr / sqrt(E[g^2]_t + eps)) * g_t
        """
        lr = 0.01
        rho = 0.9
        eps = 1e-8
        opt = RMSprop(lr=lr, rho=rho, epsilon=eps)

        p = np.array([2.0, -1.0], dtype=np.float64)
        g1 = np.array([0.4, -0.2], dtype=np.float64)

        # Step 1:
        # E[g^2]_1 = (1 - 0.9) * g1^2 = 0.1 * [0.16, 0.04] = [0.016, 0.004]
        # p_1 = p_0 - (lr / sqrt(E[g^2]_1 + eps)) * g1
        opt.step([p], [g1])
        expected_eg2_1 = (1.0 - rho) * (g1 ** 2)
        expected_rms1 = np.sqrt(expected_eg2_1 + eps)
        expected_p1 = np.array([2.0, -1.0]) - (lr / expected_rms1) * g1

        np.testing.assert_allclose(opt.avg_squared_grads[0], expected_eg2_1, rtol=1e-7)
        np.testing.assert_allclose(p, expected_p1, rtol=1e-7)

        # Step 2:
        g2 = np.array([0.1, 0.5], dtype=np.float64)
        opt.step([p], [g2])
        expected_eg2_2 = rho * expected_eg2_1 + (1.0 - rho) * (g2 ** 2)
        expected_rms2 = np.sqrt(expected_eg2_2 + eps)
        expected_p2 = expected_p1 - (lr / expected_rms2) * g2

        np.testing.assert_allclose(opt.avg_squared_grads[0], expected_eg2_2, rtol=1e-7)
        np.testing.assert_allclose(p, expected_p2, rtol=1e-7)


class TestAdam(unittest.TestCase):
    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            Adam(beta1=-0.1)
        with self.assertRaises(ValueError):
            Adam(beta1=1.0)
        with self.assertRaises(ValueError):
            Adam(beta2=-0.1)
        with self.assertRaises(ValueError):
            Adam(beta2=1.0)
        with self.assertRaises(ValueError):
            Adam(epsilon=0.0)
        with self.assertRaises(ValueError):
            Adam(epsilon=-1e-6)

    def test_default_values(self):
        opt = Adam()
        self.assertEqual(opt.lr, 0.001)
        self.assertEqual(opt.beta1, 0.9)
        self.assertEqual(opt.beta2, 0.999)
        self.assertEqual(opt.epsilon, 1e-8)
        self.assertEqual(opt.t, 0)

    def test_step_counter_increments(self):
        opt = Adam()
        p = [np.array([1.0])]
        g = [np.array([0.1])]
        self.assertEqual(opt.t, 0)
        opt.step(p, g)
        self.assertEqual(opt.t, 1)
        opt.step(p, g)
        self.assertEqual(opt.t, 2)

    def test_numerical_accuracy(self):
        """
        Paper Section 4.6, Equations 19, 20, 21:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat_t = m_t / (1 - beta1^t)
        v_hat_t = v_t / (1 - beta2^t)
        theta_{t+1} = theta_t - (lr / (sqrt(v_hat_t) + eps)) * m_hat_t
        """
        lr = 0.005
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        opt = Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=eps)

        p = np.array([1.5, -2.5], dtype=np.float64)
        g1 = np.array([0.2, -0.4], dtype=np.float64)

        # Step 1 (t=1):
        opt.step([p], [g1])
        t1 = 1
        expected_m1 = (1.0 - beta1) * g1
        expected_v1 = (1.0 - beta2) * (g1 ** 2)
        expected_m_hat1 = expected_m1 / (1.0 - beta1 ** t1)
        expected_v_hat1 = expected_v1 / (1.0 - beta2 ** t1)
        expected_delta1 = - (lr / (np.sqrt(expected_v_hat1) + eps)) * expected_m_hat1
        expected_p1 = np.array([1.5, -2.5]) + expected_delta1

        np.testing.assert_allclose(opt.mt[0], expected_m1, rtol=1e-7)
        np.testing.assert_allclose(opt.vt[0], expected_v1, rtol=1e-7)
        np.testing.assert_allclose(p, expected_p1, rtol=1e-7)

        # Step 2 (t=2):
        g2 = np.array([-0.1, 0.3], dtype=np.float64)
        opt.step([p], [g2])
        t2 = 2
        expected_m2 = beta1 * expected_m1 + (1.0 - beta1) * g2
        expected_v2 = beta2 * expected_v1 + (1.0 - beta2) * (g2 ** 2)
        expected_m_hat2 = expected_m2 / (1.0 - beta1 ** t2)
        expected_v_hat2 = expected_v2 / (1.0 - beta2 ** t2)
        expected_delta2 = - (lr / (np.sqrt(expected_v_hat2) + eps)) * expected_m_hat2
        expected_p2 = expected_p1 + expected_delta2

        np.testing.assert_allclose(opt.mt[0], expected_m2, rtol=1e-7)
        np.testing.assert_allclose(opt.vt[0], expected_v2, rtol=1e-7)
        np.testing.assert_allclose(p, expected_p2, rtol=1e-7)


class TestConvergenceOnQuadratic(unittest.TestCase):
    """Verify that all optimizers converge towards minimum of f(w) = 0.5 * w^2."""
    def test_convergence(self):
        test_cases = [
            ("SGD", SGD(lr=0.1)),
            ("Momentum", Momentum(lr=0.05, momentum=0.9)),
            ("NAG", NAG(lr=0.05, momentum=0.9)),
            ("AdaGrad", AdaGrad(lr=0.5, epsilon=1e-8)),
            ("AdaDelta", AdaDelta(rho=0.9, epsilon=1e-6)),
            ("RMSprop", RMSprop(lr=0.05, rho=0.9, epsilon=1e-8)),
            ("Adam", Adam(lr=0.05, beta1=0.9, beta2=0.999, epsilon=1e-8)),
        ]

        for name, opt in test_cases:
            with self.subTest(optimizer=name):
                # Start at w = 0.5
                w = np.array([0.5], dtype=np.float64)
                for _ in range(150):
                    grad = w.copy()  # df/dw = w
                    opt.step([w], [grad])

                # After 150 steps, w should be significantly reduced towards 0 (< 0.1)
                self.assertLess(np.abs(w[0]), 0.1, f"{name} failed to converge: final w={w[0]}")


if __name__ == "__main__":
    unittest.main()
