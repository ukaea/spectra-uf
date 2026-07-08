"""
unittest: regression testing
"""

import math
import os
import unittest
import spectrauf as suf


# the directory of the test reference data
REF_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'testdata')


class CustomAssertions():
    def assertRelativeToleranceAlmostEqual(self, a, b, tol):
        msg = f'{a} and {b} are not within relative ' \
              f'tolerance {tol}'
        assert math.isclose(a, b, rel_tol=tol), msg

    def assertListsRelativeToleranceAlmostEqual(self, list_a, list_b, tol):
        for a, b in zip(list_a, list_b):
            msg = f'{a} and {b} are not within relative ' \
                  f'tolerance {tol}'
            assert math.isclose(a, b, rel_tol=tol), msg


class SPECTRAUFUnitTest5Channel(unittest.TestCase, CustomAssertions):
    def setUp(self):
        # supress logging outputs
        self.logger = suf.logger.get_logger()
        self.previous_logger_level = self.logger.getEffectiveLevel()
        self.logger.setLevel(0)

        self.unfolding_data = suf.datahandler.DataHandler()

        dir = os.path.join(REF_DATA_DIR, '5channel')

        rr = os.path.join(dir, 'reaction_rates.csv')
        rr_sig = os.path.join(dir, "reaction_rate_uncertanties.csv")
        a_priori = os.path.join(dir, "a_priori.csv")
        rm = os.path.join(dir, "response_matrix.csv")
        target = os.path.join(dir, "target.csv")

        self.unfolding_data.load('measured', rr, uncertainty=rr_sig)
        self.unfolding_data.load('a_priori', a_priori)
        self.unfolding_data.load('response_matrix', rm)
        self.unfolding_data.load('target', target)

    def tearDown(self):
        # re-set logging to previous level
        self.logger.setLevel(suf.logger.LOGGER_LEVELS.LEVEL_LOOKUP[self.previous_logger_level])

    def test_run_uf_nlsq_a_5_channel_full_rank_1e4_iterations(self):
        TARGET = [8.413065721549275, 12.553465402834425, 9.228057381553636,
                  4.989700899276409, 1.834933516650585]
        
        nlsq_a = suf.leastsquares.NonlinearLeastSquareIterativeA()
        nlsq_a.set_all(self.unfolding_data)
        nlsq_a.run(method='iterations', max_iterations=10000)

        self.assertListsRelativeToleranceAlmostEqual(nlsq_a.solution, 
                                                     TARGET,
                                                     0.01)
        
    def test_run_nlsq_a_5_channel_full_rank_reduced_chi_sq_to_1(self):
        nlsq_a = suf.leastsquares.NonlinearLeastSquareIterativeA()
        nlsq_a.set_all(self.unfolding_data)
        nlsq_a.run(method='reduced_chi_squared', reduced_chi_squared=1.0)

        target_solution = [8.84, 12.46, 8.90, 5.45, 1.55]

        self.assertListsRelativeToleranceAlmostEqual(nlsq_a.solution, 
                                                     target_solution,
                                                     0.01)
        
    def test_run_nlsq_b_5_channel_full_rank_1e4_iterations(self):
        # target solution
        SOLUTION = [8.41306572, 12.55346540, 9.22805738, 4.98970089,
                    1.83493351]

        # run nlsq_b
        nlsq_b = suf.leastsquares.NonlinearLeastSquareIterativeB()
        nlsq_b.set_all(self.unfolding_data)
        nlsq_b.run(method='iterations', max_iterations=10000)

        # assert solution equal to previous
        self.assertListsRelativeToleranceAlmostEqual(nlsq_b.solution,
                                                     SOLUTION,
                                                     1e-6)

    def test_run_nlsq_b_5_channel_full_rank_response_redcued_chi_sq_1(self):

        SOLUTION = [8.83994085, 12.46248132, 8.89716626, 5.44853739,
                    1.54713426]
        
        nlsq_b = suf.leastsquares.NonlinearLeastSquareIterativeB()
        nlsq_b.set_all(self.unfolding_data)
        nlsq_b.run('reduced_chi_squared', 1.0)

        self.assertListsRelativeToleranceAlmostEqual(nlsq_b.solution, SOLUTION, 1e-6)

    def test_run_nlsq_c_5_channel_full_rank_response_1e5_iterations(self):
        TARGET = [8.41148915, 12.55289867, 9.22656133, 4.99158727, 1.83644420]

        nlsq_c = suf.leastsquares.NonlinearLeastSquareIterativeC()
        nlsq_c.set_all(self.unfolding_data)
        nlsq_c.run('iterations', 10000)

        self.assertListsRelativeToleranceAlmostEqual(nlsq_c.solution,
                                                     TARGET,
                                                     0.01)

    def test_run_nlsq_c_5_channel_full_rank_response_chi_squared_per_degree_of_freedom_of_unity(self):
        TARGET = [8.83945283, 12.46031965, 8.90278288, 5.44748829, 1.54773515]

        nlsq_c = suf.leastsquares.NonlinearLeastSquareIterativeC()
        nlsq_c.set_all(self.unfolding_data)
        nlsq_c.run('reduced_chi_squared', 1.0)

        self.assertListsRelativeToleranceAlmostEqual(nlsq_c.solution,
                                                     TARGET,
                                                     0.10)

    def test_run_maximum_potential_entropy_full_rank_response_minimize(self):
        TARGET = [8.88173881, 12.47648042, 8.83302783, 5.47357863, 1.54439034]

        mpe = suf.entropy.MaximumPotentialEntropy()
        mpe.set_all(self.unfolding_data)
        mpe.omega = 1.0
        mpe.run(method='minimize', omega=1.0)

        self.assertListsRelativeToleranceAlmostEqual(mpe.solution, TARGET, 0.10)
