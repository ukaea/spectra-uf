"""This contains the entropy unfolding methods"""

from typing import Union
import warnings
import numpy as np
from numpy.typing import NDArray
import scipy.optimize
import matplotlib.pyplot as plt

from .checkvalues import val_passed_is_numeric, iterable_passed_is_numeric
from .datahandler import DataHandler
from .table import Table
from .leastsquares import NonlinearLeastSquareIterativeB


class MaximumPotentialEntropy(DataHandler):
    """The MaximumPotentialEntropy unfolding method is based on the underlying
    mathematics of the M. Reginatto MAXED code. Inherits from DataHandler class,
    see this for additional attributes and parameters

    Attributes
    ----------
    omega : float
        The omega value represents the number of degrees of freedom. See theory
        section of documetation for a mathematical description.
    """

    def __init__(self, *args, **kwargs):
        """intialise class"""
        super().__init__(*args, **kwargs)

        self._methods = {
            "minimize": self._run_minimize,
            "basinhopping": self._run_basinhopping,
        }

        # parameters
        self._omega: float = None
        self._lambdas: NDArray = None

        # optimisation parameters
        self._measured_tolerance: float = 0.001
        self._max_iterations: int = int(1e3)
        self._iterations: int = 0
        self._optimisation_solution: scipy.optimize.solution = None
        self._minimize_method: str = "L-BFGS-B"
        self._lambda_bounds: list[float, float] = [-1e5, 1e5]

        # method
        self._method: str = "MaximumPotentialEntropy"

    @property
    def omega(self):
        """Omega value

        Returns
        -------
        omega : float
        """
        return self._omega

    @omega.setter
    @val_passed_is_numeric
    def omega(self, omega: float):
        """Omega value

        Parameters
        ----------
        omega : float
            Sets the omga value which controls the level of fit
        """
        self._omega = omega

    @property
    def lambdas(self):
        """Lambda values

        Parameters
        ----------
        lambdas : NDArray

        """
        return self._lambdas

    @property
    def lambda_bounds(self):
        """Lambda bounds defining the upper and lower values which
        the scipy optimisation method can search within
        """
        return self._lambda_bounds

    @lambda_bounds.setter
    @iterable_passed_is_numeric
    def lambda_bounds(self, lambda_bounds: list[float, float]):
        """Lambda bounds setter, setting the upper and lower values
        whcih the scipy optimisation method can search within
        """
        self._lambda_bounds = lambda_bounds

    @property
    def optimisation_solution(self):
        """
        returns the optimisation solution produced when running the
        scipy minimize or basinhopping methods.

        Returns
        -------
        optimisation_solution : scipy.optimization.solution
            returns the scipy optimisation method solution
        """
        return self._optimisation_solution

    def _calculate_max_tolerance(
        self, spectrum: Union[list[float], np.typing.NDArray], log_measured=False
    ):
        """
        calculates the maximum divergence of the calculated measured against
        the measured values

        Parameters
        ----------
        spectrum : Union[list[float], np.typing.NDArray]
            Iterable containing a spectrum

        Returns
        -------
        fitted_tol_max_difference : float
            The maximum dovergence of the measured against the calculated
            measured values produced by the solution spectrum

        """
        # calculate measured from spectrum
        measured_calculated = np.dot(self._response_matrix, spectrum)

        # calculate maximum divergence from measured
        fitted_tol_min = min(measured_calculated / self._measured)
        fitted_tol_max = max(measured_calculated / self._measured)
        fitted_tol_max_difference = max(
            [abs(1 - fitted_tol_min), abs(fitted_tol_max - 1)]
        )

        if log_measured:
            headings: list[str] = [
                "measured",
                "calculated",
                "measured/calculated ratio",
            ]
            rows: list[list] = [
                [float(measured), float(calculated), float(measured / calculated)]
                for measured, calculated in zip(self._measured, measured_calculated)
            ]
            table: Table = Table(rows, headers=headings)
            self.logger.info(table.string)

        return fitted_tol_max_difference

    def run(
        self,
        method: str = "basinhopping",
        omega: float = None,
        measured_tolerance: float = 0.02,
        max_iterations: int = int(1e3),
        minimize_method: str = "L-BFGS-B",
    ):
        """Run method

        Parameters
        ----------
        method : str (optional)
            The run methods requested, can be basinhopping or minimize, and
            represents the optimisation method required by the user. Default
            is the minimize optimisation method, an alterative is basinhopping

        omega : float (optional)
            The omega value represents the number of degrees of freedom within
            the unfolding kernel. This defaults to the number of measured values,
            and therefore free lambda parameters to be optimised.

        measured_tolerance : float (optional)
            The measured tolerance

        max_iterations : int (optional)
            The maxiter option passed to the scipt optimisation method.
            (Default = 1e4)

        minimize_method : str (optional)
            The requested optimisation algorithm passed to the scipy minimize
            optimizer. (default = 'L-BFGS-B'). See scipy documentation for
            list of options.

        basinhopping_max_iter : int (optional)
            The maximum number of iterations the basin hopping method can
            take. A smaller number may lead to reduced serch of solution space
            of the lambda values, but also quicker run time.
        """

        # set fitting parameters
        self._measured_tolerance = measured_tolerance
        self._max_iterations = max_iterations

        # optimisation parameters
        self._minimize_method = minimize_method

        # set omega value - defaults to the number of measured values
        if isinstance(omega, type(None)):
            self._omega = self._m
        else:
            self._omega = omega

        # scale the a priori spectrum
        self.scale_a_priori()

        # set initial lambda values
        self._lambdas = np.zeros((self._m))

        # run optimisation method
        solution = self._run(method)

        # calculate solutoin from lambda values
        self._lambdas = solution

        # calculate the solution spectrum from the lambda values
        self._solution = self._convert_lambdas_to_flux(self._lambdas)

        # calculate maximum divergence of calculated to measured calues

        max_tolerance = self._calculate_max_tolerance(self._solution, log_measured=True)

        self.logger.info(
            f"maximum divergence of solution to measured: {max_tolerance:.4f}"
        )

    def _callback_minimize(self, params):
        current_solution = self._convert_lambdas_to_flux(params)
        max_divergence = self._calculate_max_tolerance(current_solution)

        self._iterations += 1
        if self._iterations % 500 == 0:
            self._calculate_max_tolerance(current_solution, log_measured=True)

        if max_divergence < self._measured_tolerance:
            raise StopIteration

        return False

    def _wrap_callback_minimize(self):
        def func(params):
            return self._callback_minimize(params)

        return func

    def _run_minimize(self):
        """run_minimize calls the scipy minimize method to optimise the
        lambda functions to maximise the relative entrop potential function

        Parameters
        ----------
        optimisation_method : str, optional
            The optimisation method used in the scipy optimisation method

        Returns
        -------
        solution : solution
            The solution of the optimisation method

        """

        # Stops the warnings being printed to screen when lambda parameters
        # are picked that cause large solution values
        warnings.filterwarnings("ignore")

        # pass run options for minimize method
        options = {"maxiter": self._max_iterations}

        # setup callback, wrapped to allow minimize access to class methods
        callback = self._wrap_callback_minimize()

        # set bounds for lambda parameters
        bounds = [self._lambda_bounds for _ in range(len(self._lambdas))]

        # run minimize
        solution = scipy.optimize.minimize(
            self._potential_function,
            self._lambdas,
            method=self._minimize_method,
            options=options,
            bounds=bounds,
            callback=callback,
        )

        return solution.x

    def _run_basinhopping(self):
        """run_basinhopping calls the scipy basinhopping method to optimise
        the lambda functions to maximise the relative entrop potential function
        """
        # Stops the warnings being printed to screen when lambda parameters
        # are picked that cause large solution values, as these sampled valuse
        # will be ignored.
        warnings.filterwarnings("ignore")

        # define local optimisation search criteria (minimize)
        minimizer_kwargs = {
            "method": self._minimize_method,
            "bounds": [self._lambda_bounds for _ in self._measured],
        }

        solution = scipy.optimize.basinhopping(
            self._potential_function,
            self._lambdas,
            niter=self._max_iterations,
            # minimizer_kwargs=minimizer_kwargs
        )

        return solution.x

    def _potential_function(self, lambdas):
        """Potential function, calculates the relative entropy potential of
        the provided lambda multipliers

        Parameters
        ----------
        lambdas : np.nd_array
            Optimisation paraneters of relative entropy potential

        Returns
        -------
        entropy_potential : float
            The relative entropy potential of the current lambda value
        """

        # calculate the potential entropy
        np_exp_sum = np.sum((lambdas * self._response_matrix.T).T, axis=0)

        # check that values do not exceed floating point maximum, else hit a ceiling
        if np_exp_sum.max() > 500:
            return 1e99

        a_priori_term = np.sum(self._a_priori * np.exp(-np_exp_sum))

        np_uncertainty_sum = np.sum((lambdas * self._measured_uncertainty) ** 2)
        uncertainty_term = (self._omega * np_uncertainty_sum) ** 0.5

        reaction_rate_term = np.sum(self._measured * lambdas)

        z = -a_priori_term - uncertainty_term - reaction_rate_term

        return -z

    def _convert_lambdas_to_flux(self, lambdas):
        """Will convert the lambda optimisation values to a solution

        Parameters
        ----------
        lambda_values : np.nd_array
            Lambda values of the relative entropy potential function

        Returns
        -------
        spectrum : np.nd_array
            The spectrum associated with the set of lambda values provided
        """

        # calculate solution spectrum
        solution = self._a_priori * np.exp(-np.dot(lambdas, self._response_matrix))

        # return solution spectrum
        return solution
