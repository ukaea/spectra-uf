import unittest
import numpy as np
import spectrauf as suf

class SPECTRAUFRegularizationUnitTest(unittest.TestCase):
    def setUp(self):
        # supress logging outputs
        self.logger = suf.logger.get_logger()
        self.previous_logger_level = self.logger.getEffectiveLevel()
        self.logger.setLevel(0)

    def tearDown(self):
        # re-set logging to previous level
        self.logger.setLevel(suf.logger.LOGGER_LEVELS.LEVEL_LOOKUP[self.previous_logger_level])

    def test_tikhonov_identity(self):
        """With lambda=0, the output should match the input."""
        data = np.array([10.0, 5.0, 2.0, 7.0])
        smoothed = suf.regularization.tikhonov_smoothing(data, lambda_param=0.0)
        np.testing.assert_array_almost_equal(data, smoothed, decimal=6)

    def test_tikhonov_positivity_constraint(self):
        """Check that the solver enforces the x >= 0 bound even with negative input."""
        data = np.array([-10.0, 5.0, -2.0, 7.0])
        smoothed = suf.regularization.tikhonov_smoothing(data, lambda_param=0.1)
        self.assertTrue(np.all(smoothed >= 0))

    def test_tikhonov_smoothing_effect(self):
        """Check that smoothing actually reduces curvature."""
        # A very 'jagged' signal
        jagged = np.array([1.0, 10.0, 1.0, 10.0, 1.0])
        smoothed = suf.regularization.tikhonov_smoothing(jagged, lambda_param=1.0, differential=2)
        
        # Curvature (2nd derivative) of smoothed should be less than original
        orig_curv = np.sum(np.abs(np.diff(jagged, n=2)))
        smooth_curv = np.sum(np.abs(np.diff(smoothed, n=2)))
        
        self.assertLess(smooth_curv, orig_curv)

if __name__ == '__main__':
    unittest.main()
