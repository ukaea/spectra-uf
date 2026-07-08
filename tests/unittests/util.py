import spectrauf as suf
from spectrauf.util import kullback_leibler_divergence
import unittest
import numpy as np


class SPECTRAUFUtilUnitTest(unittest.TestCase):
    def setUp(self):
        # supress logging outputs
        self.logger = suf.logger.get_logger()
        self.previous_logger_level = self.logger.getEffectiveLevel()
        self.logger.setLevel(0)

        # mock up histogram
        self.bounds = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        self.hist = np.array([10.0, 20.0, 30.0, 20.0, 10.0])
        
        # mock vectors to test dkl
        self.vec_p = np.array([0.2, 0.5, 0.3])
        self.vec_q = np.array([0.1, 0.6, 0.3])
        self.err = np.array([0.02, 0.05, 0.03])

    def tearDown(self):
        # re-set logging to previous level
        self.logger.setLevel(suf.logger.LOGGER_LEVELS.LEVEL_LOOKUP[self.previous_logger_level])

    def test_dkl(self):
        p = np.array([9/25, 12/25, 4/25])
        q = np.array([1/3, 1/3, 1/3])

        dkl = kullback_leibler_divergence(p, q)

        self.assertAlmostEqual(dkl, 0.0852996, 5)

    def test_dkl_requires_numpy_or_list_or_tuple(self):
        p = [9/25, 12/25, 4/25]
        q = 'some_string'

        msg = f'kullback_leibler_divergence requires 1d' \
              f'iterable or numpy array'

        self.assertRaises(Exception, kullback_leibler_divergence, p, q)

    def test_rebin_linear_conservation(self):
        """Test if linear rebinning conserves the total integral."""
        # Rebin to 3 larger bins
        new_bounds = np.array([0.0, 4.0, 7.0, 10.0])
        rebinned = suf.util.rebin(self.hist, self.bounds, new_bounds, method="linear")
        
        self.assertAlmostEqual(np.sum(self.hist), np.sum(rebinned), places=5)
        # Check specific bin calculation: First new bin [0-4] should be 10 + 20 = 30
        self.assertEqual(rebinned[0], 30.0)

    def test_rebin_log_method(self):
        """Test log-space rebinning doesn't crash and remains ascending."""
        log_bounds = np.array([1e-6, 1e-3, 1.0, 10.0])
        log_hist = np.array([100, 50, 10])
        new_log_bounds = np.array([1e-5, 1e-1, 5.0])
        
        rebinned = suf.util.rebin(log_hist, log_bounds, new_log_bounds, method="log")
        self.assertEqual(len(rebinned), len(new_log_bounds) - 1)

    def test_lethargy_round_trip(self):
        """Test that converting to lethargy and back returns the original flux."""
        # Avoid 0 in bounds for log calculation
        bounds = np.array([1.0, 2.0, 3.0, 4.0])
        flux = np.array([10.0, 15.0, 20.0])
        
        lethargy = suf.util.convert_spectrum_to_lethargy(flux, bounds)
        back_to_flux = suf.util.convert_lethargy_to_spectrum(lethargy, bounds)
        
        np.testing.assert_array_almost_equal(flux, back_to_flux)

    def test_kullback_leibler_divergence(self):
        """Test KL Divergence with known small values."""
        div = suf.util.kullback_leibler_divergence(self.vec_p, self.vec_q)
        self.assertGreater(div, 0) # Divergence is always positive for non-identical vectors
        
        # Test identity
        div_zero = suf.util.kullback_leibler_divergence(self.vec_p, self.vec_p)
        self.assertAlmostEqual(div_zero, 0.0)

    def test_temperature_conversions(self):
        """Test Kelvin and Centigrade to MeV."""
        # 20°C is ~293.15K. At 293.15K, E is ~2.5e-8 MeV (Thermal)
        mev_c = suf.util.centigrade_to_mev(20.0)
        mev_k = suf.util.kelvin_to_mev(293.15)
        
        self.assertAlmostEqual(mev_c, mev_k, places=10)
        self.assertAlmostEqual(mev_k, 2.527e-8, delta=1e-10)

    def test_strip_zeros(self):
        """Test leading and trailing zero removal."""
        v = np.array([0, 0, 1, 2, 0, 3, 0, 0])
        trimmed, start, end = suf.util.strip_leading_or_trailing_zeros(v)
        
        np.testing.assert_array_equal(trimmed, [1, 2, 0, 3])
        self.assertEqual(start, 2)
        self.assertEqual(end, 6)

    def test_reduced_chi_squared(self):
        """Test reduced chi squared calculation."""
        # If model matches measured exactly, chi_sq should be 0
        chi = suf.util.reduced_chi_squared(self.vec_p, self.vec_p, self.err)
        self.assertEqual(chi, 0.0)
        
        # Simple manual check
        model = np.array([10.0])
        meas = np.array([8.0])
        err = np.array([2.0])
        # ((10-8)/2)^2 / 1 = 1.0
        self.assertEqual(suf.util.reduced_chi_squared(model, meas, err), 1.0)

if __name__ == '__main__':
    unittest.main()