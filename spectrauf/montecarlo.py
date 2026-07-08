"""Monte Carlo method for condition analysis and error propogation"""

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfinv

from .datahandler import DataHandler
from .logger import setup_logging, LOGGER_LEVELS


class MonteCarlo(DataHandler):
    """MonteCarlo is a class that runs a given unfolding method whilst
    stochastically varying the input to allow for quantification of uncertainty
    or estimation of condition for a given unfolding problem.
    """

    def __init__(self, *args, **kwargs):
        """Initialise class"""
        super().__init__(*args, **kwargs)

        # set up logging
        self.logger = setup_logging()

        # parameters for unfolding method
        self._unfolding_method: DataHandler = None
        self._unfolding_method_params: dict = None

        # parameters for monte carlo method
        self._min_trials: int = None
        self._tolerance: float = None
        self._confidence: float = None
        self._batch_size: int = None
        self._trials: int = 0
        self._batches: int = 0
        self._batches_run: int = 0
        self._delta: float = None

        # monte carlo solution
        self._mc_solutions: NDArray = None
        self._mc_solutions_mean: NDArray = None
        self._mc_solution_std_dev: NDArray = None

        # solution
        self._unfolded_data: NDArray = None

    @property
    def batches(self):
        """Gets the number of batches that have been run

        Returns
        -------
        batches : int
            The number of batches that have been run
        """

        return self._batches

    @property
    def confidence(self):
        """Gets the confidence

        Returns
        -------
        confidence : float
            The confidence interval of the Monte Carlo oncvergence criteria
        """

        return self._confidence

    @property
    def unfolded_data(self):
        """Gets the solution

        Returns
        -------
        solution : np.nd_array
            The solution
        """
        return self._unfolded_data

    @property
    def monte_carlo_solution(self):
        """Gets the mean monte carlo solution

        Returns
        -------
        monte_carlo_solutions_mean : np.nd_array
            The mean of the monte carlo solution spectra across all solutions generated
        """

        return np.mean(self._mc_solutions, axis=0)

    @property
    def tolerance(self):
        """Gets the tolerance

        Returns
        -------
        tolerance : float
            The tolerance of the Monte Carlo convergence criteria
        """

        return self._tolerance

    @property
    def trials(self):
        """Gets the number of trials that have been run

        Returns
        -------
        trials : int
            The number of trials that have been run
        """

        return self._trials

    @property
    def monte_carlo_solution_std_dev(self):
        """Gets the solution standard deviations

        Returns
        -------
        solution_std_dev : np.nd_array
            The standard deviations of th solutoin mean
        """
        return np.std(self._mc_solutions, axis=0)

    @property
    def solutions(self):
        """returns solutions from all monte carlo trials
        in a 2D array

        returns
        -------
        mc_solutions : NDArray
            The 2D array storing all of the solutions from
            all monte carlo trials
        """
        return self._mc_solutions

    def run(
        self,
        unfolding_method: DataHandler,
        run_method: str,
        run_parameters: dict,
        min_trials: int = 100,
        tolerance: float = 0.1,
        confidence: float = 0.95,
        batch_size: int = 5,
        parameters_to_randomise=None,
    ):
        """Run method

        Parameters
        ----------
        run_type : Unfolding method (child of DataHandler)
            The unfolding method to be run
        run_parameters : dict (key word arguments for run method)
            Key word run method parameters which are passed to the run method
        min_trials : int, optional
            The minimum number of trials to be run before the convergence
            criteria will be chekced.
        tolerance : float, optional
            The tolerance of the convergence criteria
        confidence : float, optional
            The confidence interval to be used by the convergence criteria
        batch_size : int, optional
            The number of trials to be run in each batch. The convergence
            criteria is only checked at the end of each batch
        parameters_to_randomise : list, optional
            List of strings describing the variables to be randomised in the
            Monte Carlo method
        n_threads: int, optional
            The number of threads used in the multiprocessing pool
        """

        # unfolding method
        self._unfolding_method = unfolding_method

        # run parameters
        self._min_trials = min_trials
        self._tolerance = tolerance
        self._confidence = confidence
        self._batch_size = batch_size
        self._delta = np.sqrt(2) * erfinv(2 * (confidence / 2))
        self._mc_solutions = np.empty((0, self.shape_n))

        if isinstance(parameters_to_randomise, type(None)):
            # if unspecified find all input variables with uncertainry attached
            randomise = [
                attr
                for attr in self._settable_attributes
                if f"_{attr}_uncertainty" in self.__dict__
            ]
            randomise = [
                attr
                for attr in randomise
                if not isinstance(getattr(self, f"_{attr}_uncertainty"), type(None))
            ]
        else:
            randomise = parameters_to_randomise
            # add in here the checks to make sure there is uncertainty
            # attached to the speicifed variables

        # run Monte Carlo method
        _trials_req = 1e99

        # run a single trial
        def _run_trial():
            # set logging level to reduce output to terminal for multiple runs
            previous_logger_level = LOGGER_LEVELS.LEVEL_LOOKUP[
                self.logger.getEffectiveLevel()
            ]
            self.logger.setLevel(LOGGER_LEVELS.warning)

            data = self._unfolding_method()

            data.set_all(self)

            # randomise parameters with uncertainties
            for label in randomise:
                quantity = getattr(data, label)
                uncertainty = getattr(data, f"_{label}_uncertainty")
                covariance = getattr(data, f"_{label}_covariance")

                # randomise input variables
                setattr(data, label, np.random.normal(loc=quantity, scale=uncertainty))

            # run instance
            data.run(method=run_method, **run_parameters)

            # reset logging leve to previously set
            self.logger.setLevel(previous_logger_level)

            # return instance
            return data

        # run trials until convergence criteria is not met
        self._batches_run = 0

        while (self._trials < self._min_trials) or (self._trials < _trials_req):

            self.logger.info(
                f"[batch {self._batches_run+1}] running {self._batch_size} trials"
            )

            # run a batch
            for _ in range(self._batch_size):
                # run individual trial
                data = _run_trial()

                # add solution to the stack of solution spectrums
                self._mc_solutions = np.vstack((self._mc_solutions, data.solution))

                # add to trial count
                self._trials += 1

            self.logger.info(f" - {self._trials} trials run")
            self.logger.info(f" - {_trials_req} required trials estimated")

            self._batches_run += 1

            # calculate statistics
            # self._mc_solutions_mean = np.mean(self._mc_solutions, axis=0)
            # self._mc_solution_std_dev = np.std(self._mc_solutions, axis=0)

            _trials_req = self._trials_required()

        self.logger.info("summary")
        self.logger.info("-------")
        self.logger.info(f"tolerance: {self._tolerance}")
        self.logger.info(f"confidence: {self._confidence}")
        self.logger.info(f"batches: {self._batches_run}")
        self.logger.info(f"min trials: {self._min_trials}")
        self.logger.info(f"trials run: {self._trials}")
        self.logger.info(f"trials req.: {self._trials_required()}")

        # run the unfolding method a single time
        data = self._unfolding_method()
        data.set_all(self)
        data.run(method=run_method, **run_parameters)
        self._unfolded_data = data

    def _trials_required(self):
        """_trials_required will calculate the minimum number of trials
        required
        """

        # return np.max(np.ceil((self._delta**2) * ((self._mc_solution_std_dev**2) \
        #      / ((self._tolerance * self._mc_solutions_mean)**2))))

        return np.max(
            np.ceil(
                (self._delta**2)
                * (
                    (self.monte_carlo_solution_std_dev**2)
                    / ((self._tolerance * self.monte_carlo_solution) ** 2)
                )
            )
        )
