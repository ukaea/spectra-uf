"""Non-linear least square unfolding methods"""

import numpy as np
from spectrauf.datahandler import DataHandler


class NonlinearLeastSquareIterativeA(DataHandler):
    """NonlinearLeastSquareIterativeA is a non-linear least square gradient
    descent method based on the underlying mathematics of the USAF SAND-II code
    written in 1967.
    """

    def __init__(self, *args, **kwargs):
        """Initialise class"""
        super().__init__(*args, **kwargs)

        # output to logger
        if self.__class__ is not NonlinearLeastSquareIterativeA:
            self.logger.info("Instance created of %s", self.__class__.__name__)

        # define run methods
        self._methods = {
            "reduced_chi_squared": self._run_chi_squared,
            "iterations": self._run_iterations,
        }

        # count iterations
        self._iterations = 0

        # method vectors and matrices
        self._weights = None
        self._lambdas = None
        self._measured_iter = None
        self._ln_measured = None
        self._ln_solution = None
        self._sum_ln_z_difference = None
        self._measured_iter = None
        self._m_range = None
        self._n_range = None

        # run parameters
        self._max_iterations = np.inf
        self._target_reduced_chi_squared = 1.0

        # set method
        self._method = "NonlinearLeastSquareIterativeA"

    def run(
        self,
        method="reduced_chi_squared",
        max_iterations=np.inf,
        reduced_chi_squared=1.0,
        scale_a_priori=True,
    ):
        """Run method

        Parameters
        ----------
        method : str
            The run methods requested, can be iterations for a given number of
            iterations, .run('iterations', 100), or reduced chi squared to run
            until the chi-squared per degree of freedom falls below a given
            value, .run('reduced_chi_squared', 1.0).

        max_iterations : int
            The maximum number of iterations which will be undertaken.

        reduced_chi_squared : int
            The convergence criteria, when the reduced chi squared
            of the theoretical calculated and measured values falls below
            this value. The default = 1.0.

        scale_a_priori : bool
            Scale the a priori spectrum to match the measured values,
            the default = true.
        """

        # scale the a priori if required
        if scale_a_priori:
            self.scale_a_priori()

        # check if run method is suitable
        self._max_iterations = max_iterations
        self._target_reduced_chi_squared = reduced_chi_squared

        # run method specific things
        if self._iterations == 0:
            self._weights = np.zeros((self._m, self._n), dtype=self._np_dtype)
            self._lambdas = np.zeros((self._n), dtype=self._np_dtype)
            self._measured_iter = np.zeros((self._m), dtype=self._np_dtype)
            self._ln_measured = np.log(self._measured, dtype=self._np_dtype)
            self._sum_ln_z_difference = np.zeros((self._n), dtype=self._np_dtype)
            self._solution = np.array(self._a_priori, dtype=self._np_dtype)
            self._ln_solution = np.log(self._solution, dtype=self._np_dtype)
            self._m_range = range(self._m)
            self._n_range = range(self._n)

        self._run(method)

    def _run_chi_squared(self):
        """_run_reduced_chi_squared will run interations of the method until
        a reduced chi-squared is reached.

        Parameters
        ----------
        target_reduced_chi_sq : float
            The target chi-squared per degree of freedom
        """

        if self._iterations == 0:
            self.logger.info(
                f"iteration: {self._iterations:< 6}"
                f"reduced chi-squared: {self.reduced_chi_squared:.2f}"
            )

        while (
            self.reduced_chi_squared > self._target_reduced_chi_squared
            and self._iterations < self._max_iterations
        ):
            self._run_iteration()

    def _run_iterations(self):
        """_run_iterations will run a given number of iterations

        Parameters
        ----------
        iterations
            The required number of iterations to run
        """

        if self._iterations == 0:
            self.logger.info(
                f"iteration: {self._iterations:< 6}"
                f"reduced chi-squared: {self.reduced_chi_squared:.2f}"
            )

        while self._iterations < self._max_iterations:
            self._run_iteration()

    def _run_iteration(self):
        """_run_iteration run a single iteration of the unfolding algorithm"""
        self._measured_iter = np.dot(self._response_matrix, self._solution)

        self._weights = (
            self._response_matrix.T / self._measured_iter
        ).T * self._solution.T

        # calculate lambda values
        self._calculate_lambdas()

        # calculate solution spectrum
        self._calculate_solutions()

        # increase iteration count
        self._iterations += 1

        # log current iteration with chi-squared
        self.logger.info(
            f"iteration: {self._iterations:< 6} reduced chi-squared: {self.reduced_chi_squared:.2f}"
        )
        self.logger.debug(f"solution: {self._solution}")

    def _calculate_lambdas(self):
        """_calculate_lambdas calculates the lambda multipliers at the current
        iterative step
        """
        self._lambdas = np.sum(self._weights, axis=0) ** -1

    def _calculate_solutions(self):
        """_calculate_solutions will calculate the solution spectrum from
        the lambda parameters
        """
        m_delta = self._ln_measured - np.log(self._measured_iter)

        sum_ln_measured_difference = sum((m_delta * self._weights.T).T, 0)
        self._ln_solution += self._lambdas * sum_ln_measured_difference
        self._solution = np.exp(self._ln_solution)

    @property
    def iterations(self):
        """The number of iterations undertaken by the algorithm
        returns
        -------
        iterations : int
            The number of itrations undertaken by the algorithm
        """
        return self._iterations


class NonlinearLeastSquareIterativeB(NonlinearLeastSquareIterativeA):
    """NonlinearLeastSquareIterativeB is a non-linear least square gradient
    descent method based on the underlying mathematics of the GRAVEL and
    SAND-II algorithms, written by the USAF and M. Matzke respectively.
    """

    def __init__(self, *args, **kwargs):
        """Initialise class"""
        super().__init__(*args, **kwargs)

        # output to logger
        if self.__class__ is not NonlinearLeastSquareIterativeB:
            self.logger.info("Instance created of %s", self.__class__.__name__)

        self._method = "NonlinearLeastSquareIterativeB"

    def _calculate_lambdas(self):
        """_calculate_lambdas calculates the lambda multipliers at the current
        iterative step
        """
        for j in range(self._n):
            self._lambdas[j] = 0.0
            for i in range(self._m):
                self._lambdas[j] += (
                    self._weights[i][j]
                    / (self.measured_uncertainty[i] / self.measured[i]) ** 2
                )
            self._lambdas[j] = self._lambdas[j] ** -1

    def _calculate_solutions(self):
        """_calculate_solutions will calculate the solution spectrum from
        the lambda parameters
        """
        for j in range(self._n):
            sum_ln_measured_difference = 0.0
            for i in range(self._m):
                sum_ln_measured_difference += (
                    (self._ln_measured[i] - np.log(self._measured_iter[i]))
                    * self._weights[i][j]
                    / (self.measured_uncertainty[i] / self.measured[i]) ** 2
                )
            self._ln_solution[j] += self._lambdas[j] * sum_ln_measured_difference
            self._solution[j] = np.exp(self._ln_solution[j])


class NonlinearLeastSquareIterativeC(DataHandler):
    """NonlinearLeastSquareIterativeC is a non-linear least square gradient
    descent method based on the underlying mathematics of the SPUNIT
    algorithms, inherits from DataHandler, see for all attributes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._iterations = 0

        self._methods = {
            "reduced_chi_squared": self._run_chi_squared,
            "iterations": self._run_iterations,
        }

        # run parameters
        self._max_iterations = np.inf
        self._target_reduced_chi_squared = 1.0
        self._weight_initial_measured = None

        self._method = "NonlinearLeastSquareIterativeC"

    def run(
        self,
        method="reduced_chi_squared",
        max_iterations=np.inf,
        reduced_chi_squared=1.0,
        scale_a_priori=True,
    ):
        """Run method

        Parameters
        ----------
        method : str
            The run methods requested, can be iterations for a given number of
            iterations, .run('iterations', 100), or reduced chi squared to run
            until the chi-squared per degree of freedom falls below a given
            value, .run('reduced_chi_squared', 1.0).

        max_iterations : int
            The maximum number of iterations which will be undertaken.

        reduced_chi_squared : int
            The convergence criteria, when the reduced chi squared
            of the theoretical calculated and measured values falls below
            this value. The default = 1.0.

        scale_a_priori : bool
            Scale the a priori spectrum to match the measured values,
            the default = true
        """

        # scale the a priori if required
        if scale_a_priori:
            self.scale_a_priori()

        # check if run method is suitable
        self._max_iterations = max_iterations
        self._target_reduced_chi_squared = reduced_chi_squared

        if self._iterations == 0:
            self._solution = np.array(self._a_priori, dtype=self._np_dtype)
            self._weight_initial_measured = (
                self._response_matrix / self._measured[:, None]
            ).sum(axis=0)

        self._run(method)

    def _run_chi_squared(self):
        """_run_reduced_chi_squared will run interations of the method until
        a reduced chi-squared is reached.

        Parameters
        ----------
        target_reduced_chi_sq : float
            The target chi-squared per degree of freedom
        """

        if self._iterations == 0:
            self.logger.info(
                f"iteration: {self._iterations:< 6}"
                f"reduced chi-squared: {self.reduced_chi_squared:.2f}"
            )

        while (
            self.reduced_chi_squared > self._target_reduced_chi_squared
            and self._iterations < self._max_iterations
        ):
            self._run_iteration()

    def _run_iterations(self):
        """_run_iterations will run a given number of iterations

        Parameters
        ----------
        iterations
            The required number of iterations to run
        """
        if self._iterations == 0:
            self.logger.info(
                f"iteration: {self._iterations:< 6} reduced chi-squared: {self.reduced_chi_squared:.2f}"
            )

        while self._iterations < self._max_iterations:
            self._run_iteration()

    def _run_iteration(self):
        """_run_iteration run a single iteration of the unfolding algorithm"""

        # calculate the measured for the previous iteration
        measured_at_iteration = np.dot(self._response_matrix, self._solution)

        # calculate the weight for this iteration
        weight_for_iteration = (
            self._response_matrix / measured_at_iteration[:, None]
        ).sum(axis=0)

        self._solution = (
            self._solution * weight_for_iteration / self._weight_initial_measured
        )

        self.logger.info(
            f"iteration: {self._iterations + 1:< 6} reduced chi-squared: {self.reduced_chi_squared:.2f}"
        )

        self.logger.debug(f"solution: {self._solution}")

        self._iterations += 1


class NonLinearLeastSquareD(DataHandler):
    """NonLinearLeastSquareD is based on the underlying mathematics of the
    STAY'SL PNNL unfolding algorithm.
    """

    def __init__(self, *args, **kwargs):
        """Initalise class"""
        super().__init__(*args, **kwargs)

        self._iterations = 0

        self._methods = {
            "reduced_chi_squared": self._run_chi_squared,
            "iterations": self._run_iterations,
        }

        # run parameters
        self._max_iterations = np.inf
        self._target_reduced_chi_squared = 1.0

        self._method = "NonLinearLeastSquareD"

    def run(
        self,
        method="reduced_chi_squared",
        max_iterations=np.inf,
        reduced_chi_squared=1.0,
    ):
        """Run method

        Parameters
        ----------
        method : str
            The run methods requested, can be iterations for a given number of
            iterations, .run('iterations', 100), or reduced chi squared to run
            until the chi-squared per degree of freedom falls below a given
            value, .run('reduced_chi_squared', 1.0).

        *args : float
            The arguments associated with the run method, either n iterations
            if running 'iterations', or the reduced chi-squared target if
            runnign 'reduced-chi squared'.
        """

        # check if run method is suitable
        self._max_iterations = max_iterations
        self._target_reduced_chi_squared = reduced_chi_squared

        if self._iterations == 0:
            # set up initial vectors and matrices here
            # self._c = self._response_matrix * self._a_priori

            # calcualte relative covaraince of the a priori spectrum
            # self._ap_rel_cov = crc(self._a_priori, self.a_priori_covariance)
            pass

        self._run(method)

    def _run_chi_squared(self):
        """_run_reduced_chi_squared will run interations of the method until
        a reduced chi-squared is reached.

        Parameters
        ----------
        target_reduced_chi_sq : float
            The target chi-squared per degree of freedom
        """
        while (
            self.reduced_chi_squared > self._target_reduced_chi_squared
            and self._iterations < self._max_iterations
        ):
            self._run_iteration()

    def _run_iterations(self):
        """_run_iterations will run a given number of iterations

        Parameters
        ----------
        iterations
            The required number of iterations to run
        """
        while self._iterations < self._max_iterations:
            self._run_iteration()

    def _run_iteration(self):
        """ """
        self.logger.info(f"iteration {self._iterations + 1}")

        self.logger.debug(f"solution: {self._solution}")

        self._iterations += 1
