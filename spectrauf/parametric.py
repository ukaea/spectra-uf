"""Parametric unfodling methods"""

import math
import copy
from typing import Union, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, basinhopping

from .constants import BOLTZMANN, ELEMENTARY_CHARGE
from .util import convert_spectrum_to_lethargy as spec2leth
from .util import convert_lethargy_to_spectrum as leth2spec
from .datahandler import DataHandler


def _amplitude_pdf(energies: Union[list[float], NDArray], amplitude: float):
    """returns a flat spectrum at a given amplitude

    parameters
    ----------
    energies : NDArray
        A vector containing the energies at which to sample points
    ammplitude : float
        The amplitude by which to scale the sampled values

    returns
    -------
    pdf : NDArray
        The probability density function of the distribution
    """
    # calculate the pdf
    pdf: NDArray = np.ones(energies.shape) * amplitude

    # return pdf
    return pdf


def _log_slope_pdf(energies, log_gradient, e_min, e_max):
    """return a straight line in log space, with a gradient in
    log space between a given energy range

    parameters
    ----------
    energies : NDArray
        A vector containing the energies at which to sample points
    log_gradient : float
        The gradient of the straight line in log space
    e_min : float
        The minimum energy to allow samples to be taken
    e_max : float
        The maximum energy to allow samples to be taken

    returns
    -------
    pdf : NDArray
        The probability density function of the distribution
    """
    ys = np.zeros(energies.shape)

    for i, energy in enumerate(energies):
        if e_min <= energy <= e_max:
            ys[i] = energy**log_gradient

    return ys


def _gaussian_pdf(energies, mu, sigma):
    """returns the probability density function (pdf) of a
    gaussian distribution

    parameters
    ----------
    energies : NDArray
        A vector containing the energies at which to sample points
    mu : float
        The ceentroid of the gaussian pdf
    sigma : float
        The standard deviation (sigma) of the gaussian pdf

    returns
    -------
    pdf : NDArray
        The probability density function of the distribution

    """
    return (1.0 / np.sqrt(2 * np.pi * (sigma**2))) * np.exp(
        -((energies - mu) ** 2) / (2 * (sigma**2))
    )


def _log_normal_pdf(energies, mode, sigma):
    """returns the probability density function (pdf) of a log
    normal distribution

    parameters
    ----------
    energies : NDArray
        A vector containing the energies at which to sample points
    mode : float
        The mode of the log normal distrobution
    sigma : float
        The standard deviation of the log normal distribution

    returns
    -------
    pdf : NDArray
        The probability density function of the distribution

    """
    mu = np.log(mode) + (sigma**2)
    return (1 / (energies * sigma * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(energies) - mu) ** 2) / (2 * (sigma**2))
    )


def _maxwellian_pdf(energies, temperature_in_kelvin):
    """return the probability density function (pdf) of a maxwellian
    distribution

    parameters
    ----------
    energies : NDArray
        A vector containing the energies at which to sample points
    temperature_in_kelvin : float
        The temperature of the maxwellian distribution in kelvin

    returns
    -------
    pdf : NDArray
        The probability density function of the distribution

    """
    # convert energy into joules from MeV
    energy_in_joules = energies * ELEMENTARY_CHARGE * 1e6

    # calculate maxwellian pdf
    return (
        (2 * ((1 / math.pi) ** (1 / 2)))
        * (energy_in_joules ** (1 / 2))
        * (1 / ((BOLTZMANN * temperature_in_kelvin) ** (3 / 2)))
        * np.exp(-energy_in_joules / (BOLTZMANN * temperature_in_kelvin))
    )


def _watt_spectrum_pdf(energies, a, b):
    """return the probability density function (pdf) of a watt spectrum
    distribution in the form of: f(E) = exp(-E/a)sinh(sqrt(bE))

    parameters
    ----------
    energies : NDArray
        A vector containing the energies at which to sample points
    a : float
        The exponent denominator in MeV
    b : float
        The hyperbolic sin energy coefficient in MeV

    returns
    -------
    pdf : NDArray
        The probability density function of the distribution

    """
    # calculate the watt spectrum
    return np.exp(-energies / a) * np.sinh(np.sqrt(b * energies))


class Distribution(DataHandler):
    """virtual base class used as template for parametric distributions"""

    def __init__(self, *args, **kwargs):
        # ititalise parent
        super().__init__(*args, **kwargs)

        # define distribution
        self._amplitude: float = None
        self._amplitude_range: list[float, float] = []
        self._params: list[float] = []
        self._param_ranges: list[float] = []
        self._n_params: int = None
        self._type: str = None
        self._interpolation_scale: str = "log"

        # variable to store callable method which will return sampalable distribution
        self._distribution: Callable = None

    @property
    def sample(self):
        """samples the pdf of the distribution

        returns
        -------
        sample : NDArray
            An array of values sampled from the distribution using
            the group structure provided.

        """
        # sample distribution at group structure bin mid points and scale by bin width
        return self.distribution(*self._params)

    @property
    def sample_per_unit_lethargy(self):
        """samples the pdf of the distribution and returns the spectrum
        in flux per unit lethargy.

        returns
        -------
        sample_per_unit_lethargy : NDARray
            An array of the values samples from the distribution in per
            unit lethargy using the group structure provided

        """
        # sample distribution at group structure bin mids and scale by bin width
        pdf_scaled = self.distribution(*self._params)

        # return the spectrum scaled to per unit lethargy
        return spec2leth(pdf_scaled, self._group_structure)

    @property
    def param_ranges(self):
        """The ranges of all free parameters which are passed to the optimisation
        method

        returns
        -------
        param_ranges : NDArray
            A 2D array of the free parameter ranges, with pairs of min/max values
            in the form [[x1_min, x1_max], ..., [xn_min,xn_max]]

        """
        return self._param_ranges

    @property
    def amplitude(self):
        """The amplitude applied to the current distribution

        Returns:
            amplitude : float
                The amplitude of the current distribution
        """
        return self._amplitude

    @property
    def amplitude_range(self):
        """The range of the amplitude applied to the current distribution

        Returns:
            amplitude_range : NDArray
        """
        return self._amplitude_range

    @property
    def type(self):
        """A string describing the currnet distribution type

        Returns:
            type : str
                A string of the current distribution type
        """
        return self._type

    def scale_amplitude(self, scale_factor: float):
        """A method to scale the amplitude of the current distribution along with
        the range of possible values the optimisation method can sample within

        Parameters
        ----------
        scale_factor : float
            The scale factor by which the amplitude and the range of the amplitude
            are scaled by, typically used when scaling the a priori distributions
            to the measured values
        """

        # scale the amplitude
        self._amplitude *= scale_factor
        self._params[0] *= scale_factor

        # scale the amplitude range
        self._amplitude_range[0] = self._amplitude_range[0] * scale_factor
        self._amplitude_range[1] = self._amplitude_range[1] * scale_factor

        self._param_ranges[0][0] = self._amplitude_range[0]
        self._param_ranges[0][1] = self._amplitude_range[1]

    @property
    def bin_mid_points(self):
        """This returns the mid points across the provided group structure in
        either linear or log space

        Returns
        -------
        bin_mid_points : NDArray
            An array of the mid points taken from the group structure provided
        """

        # calculate mid points of bins to sample pdfs
        if self._interpolation_scale == "lin":
            return (self._group_structure[:-1] + self._group_structure[1:]) / 2.0
        elif self._interpolation_scale == "log":
            return 10 ** (
                (
                    np.log10(self._group_structure[:-1])
                    + np.log10(self._group_structure[1:])
                )
                / 2.0
            )
        else:
            msg = f"interpolation_scale should be either lin or log not {self._iterpolation_scale}"
            raise ValueError(msg)

    @property
    def bin_widths(self):
        """The widths of the bins in teh group structure

        Returns
        -------
        bin_widths : NDArray
            An array of the bin widths
        """
        # calcualte the bin widths
        return self._group_structure[1:] - self._group_structure[:-1]

    def __add__(self, other: Union["ParametricSpectrum", "Distribution"]):
        """Addition of the distribution with either another distribution or
        a parametric spectrum, producing a parametric spectrum.

        Parameters
        ----------
        other : Union[ParametricSpectrumt, Distribution]
            Either a ParametricSpectrum instance or a Distribution (or child)
            instance to be summed in order to produce a ParametricSpectrum with
            multiple distributions.

        Returns
        -------
        parametric_spectrum : ParametricSpectrum
            A parametric spectrum prduced by the summation of multiple distributions

        Raises
        ------
        Exception
            A type exception is raised if the other instance is not a ParametricSpectrum
            or Dicstribution (or child) instance
        """
        if isinstance(other, ParametricSpectrum):
            return ParametricSpectrum([self, *other.distributions_a_priori])
        elif isinstance(other, Distribution):
            return ParametricSpectrum([self, other])
        else:
            msg = (
                "[ERROR] - addition of Parameter and Parameter/ParametricSpectra "
                f"only allowed, not type {type(other)}"
            )
            raise TypeError(msg)


class AmplitudeDistribution(Distribution):
    """A distribution providing a flat spectrum"""

    def __init__(
        self,
        amplitude: float = 1.0,
        amplitude_range: iter = [1e-3, 1e3],
        interpolate_scale: str = "log",
    ):
        """The AmplitudeDistribution provides a flat spectrum

        Parameters
        ----------
        amplitude : float, optional
            _description_, by default 1.0
        amplitude_range : iter, optional
            _description_, by default [1e-3, 1e3]
        interpolate_scale : str, optional
            _description_, by default "log"
        """

        # initialise parent
        super().__init__()

        # initialise paremetsr
        self._amplitude: float = amplitude
        self._amplitude_range: list[float, float] = amplitude_range
        self._interpolation_scale: str = interpolate_scale

        # set type
        self._type: str = "amplitude"

        # set initial params
        self._params: list[float] = [amplitude]
        self._param_ranges: list[list[float, float]] = [self._amplitude_range]

    @property
    def distribution(self):
        """Returns a method providing the pdf of the AmplitudeDistribution which
        can be sampled with an array of energy values and a set of parameters

        returns
        -------
        distibution : Callable[NDArray, NDArray]
            A callable method which returns the pdf of the current distribution
            when provided with an array of energy positions to sample and the
            free parameters of the distribution

        """
        bin_mids = self.bin_mid_points

        # pdf method
        if not self._freeze_amplitude:

            def _amplitude_pdf_optimisable(amplitude):
                # claculate the pdf
                pdf_pul = _amplitude_pdf(bin_mids, amplitude)
                pdf = leth2spec(pdf_pul, self._group_structure)

                # return pdf
                return pdf

        else:

            def _amplitude_pdf_optimisable():
                pdf_pul = _amplitude_pdf(bin_mids, self._amplitude)
                pdf = leth2spec(pdf_pul, self._group_structure)

                # return pdf
                return pdf

        return _amplitude_pdf_optimisable

    @property
    def label(self):
        """Returns a string label describing the parameter values

        Returns
        -------
        label : str
            A string providing the distribution type and the parameter
            values of the distribution
        """
        return f"{self._type} (amp={self._amplitude:.2e})"


class SlopeDistribution(Distribution):
    """Distribution for a straight line slope in log space"""

    def __init__(
        self,
        amplitude: float = 1.0,
        amplitude_range: list[float, float] = None,
        energy_min: float = -math.inf,
        energy_max: float = math.inf,
        log_gradient: float = 1.0,
        log_gradient_range: list[float, float] = None,
        interpolation_scale: str = "log",
        freeze_gradient: bool = False,
    ):
        """A slope distribution providing a straight line in log
        space

        Parameters
        ----------
        amplitude : float, optional
            The amplitude applied to the distribution, by default 1.0
        amplitude_range : list, optional
            The range of the amplitude which the optimisation method
            can sample within, by default [1e-3, 1e3]
        energy_min : float, optional
            The minimum energy sampled, energy values provided below this
            will return zero, by default -math.inf
        energy_max : float, optional
            The maximum energy sampled, energy values provided above this
            will return zero, by default, by default math.inf
        log_gradient : float, optional
            The gradient of the distribution in log10 space, by default 1.0
        log_gradient_range : list, optional
            The range of the distribution gradient in log10 space, within which
            the optimiser can sample within, by default [-1e3, 1e3]
        interpolation_scale : str, optional
            The mid position to sample from within the goup structure, which
            can be either 'log' of 'lin' for logarithmic or linear space
            sampling respectively, by default "log"
        freeze_gradient : bool, optional
            Allows for the freezing of the gradient parameter, which will then
            remove this as a free parameter when passing distributoin to
            the optimisation method, by default False
        """

        # initialise parent
        super().__init__()

        # initialise parameters
        self._amplitude = amplitude
        self._amplitude_range = amplitude_range if not amplitude else [1e-3, 1e3]
        self._energy_min = energy_min
        self._energy_max = energy_max
        self._log_gradient = log_gradient
        self._log_gradient_range = (
            log_gradient_range if not log_gradient_range else [-1e3, 1e3]
        )
        self._interpolation_scale = interpolation_scale
        self._freeze_gradient = freeze_gradient

        # set type
        self._type = "slope"

        # set intial params
        if not self._freeze_gradient:
            self._params = [self._amplitude, self._log_gradient]
            self._param_ranges = [self._amplitude_range, self._log_gradient_range]
        else:
            raise NotImplementedError("not yet coded, sorry :(")

    @property
    def distribution(self):
        """Returns a method providing the pdf of the SlopeDistribution which
        can be sampled with an array of energy values and a set of parameters

        returns
        -------
        distibution : Callable[NDArray, NDArray]
            A callable method which returns the pdf of the current distribution
            when provided with an array of energy positions to sample and the
            free parameters of the distribution

        """
        bin_mids = self.bin_mid_points

        # pdf method
        if not self._freeze_gradient:

            def _slope_pdf_optimisable(amplitude, gradient):
                # calcualte the pdf
                pdf_pul = _log_slope_pdf(
                    bin_mids, gradient, self._energy_min, self._energy_max
                )
                pdf = leth2spec(pdf_pul, self._group_structure)

                # return normalised and scaled pdf
                return amplitude * pdf / max(pdf)

        else:
            raise NotImplementedError("not yet coded, sortry :(")

        # return distribution generator
        return _slope_pdf_optimisable

    @property
    def label(self):
        """Returns a string label describing the parameter values

        Returns
        -------
        label : str
            A string providing the distribution type and the parameter
            values of the distribution
        """
        return f"{self._type} (amp={self._amplitude:.2e})"


class MaxwellianDistribution(Distribution):
    """A maxwellian distribution"""

    def __init__(
        self,
        amplitude: float = 1.0,
        amplitude_range: list[float, float] = None,
        temperature: float = 293.15,
        temperature_range: list[float, float] = None,
        freeze_temperature: bool = False,
        interpolate_scale: str = "log",
    ):
        """A distribution providing the maxwellian pdf

        Parameters
        ----------
        amplitude : float, optional
            The amploitude applied to the normalised pdf, by default 1.0
        amplitude_range : list[float, float], optional
            The range of values of the amplitude applied to the normalsied
            pdf within which the optimistaion method can sample, by
            default [1e-2, 1e2]
        temperature : float, optional
            The temperature of the maxwellian pdf in kelvin, by default 293.15
        temperature_range : list[float, float], optional
            The range of the temperature of the maxwellian pdf in keliv
            between which the optimisation method can sample, by
            default (273.15, 2e3)
        freeze_temperature : bool, optional
            Allows for the freezing of the temperature parameter, which will then
            remove this as a free parameter when passing the distribution to
            the optimisation method, by default False
        interpolation_scale : str, optional
            The mid position to sample from within the goup structure, which
            can be either 'log' of 'lin' for logarithmic or linear space
            sampling respectively, by default "log"
        """

        # intialise parent
        super().__init__()

        # initial parameters
        self._amplitude: float = amplitude
        self._amplitude_range: list[float, float] = (
            amplitude_range if amplitude_range else [1e-2, 1e2]
        )
        self._temperature: float = temperature
        self._temperature_range: list[float, float] = (
            temperature_range if temperature_range else [273.15, 2e3]
        )
        self._freeze_temperature: bool = freeze_temperature
        self._interpolation_scale: str = interpolate_scale

        # set the amplitude to allow the peak of the mode of the maxwellian to equal the peak height
        self._mode: float = BOLTZMANN * temperature / (2.0 * ELEMENTARY_CHARGE * 1e6)

        # set type
        self._type: str = "maxwellian"

        # set intial parameters
        if not self._freeze_temperature:
            self._params = [self._amplitude, self._temperature]
            self._param_ranges = [self._amplitude_range, self._temperature_range]
        else:
            self._params = [self._amplitude]
            self._param_ranges = [self._amplitude_range]

    @property
    def params(self):
        """Returns the values of the free parameters

        Returns
        -------
        params : NDArray
            The free parameters of the distribution
        """
        return self._params

    @params.setter
    def params(self, params):
        """Sets the values of the free parameters

        Parameters
        ----------
        params : list[float, ...]
            Sets a list of the free paremters for the distribution
        """
        self._params = params

        if not self._freeze_temperature:
            self._amplitude, self._temperature = params
        else:
            self._amplitude = params[0]

    @property
    def distribution(self):
        """Returns a method providing the pdf of the SlopeDistribution which
        can be sampled with an array of energy values and a set of parameters

        returns
        -------
        distibution : Callable[NDArray, NDArray]
            A callable method which returns the pdf of the current distribution
            when provided with an array of energy positions to sample and the
            free parameters of the distribution

        """
        # get bin mid points and bin widths
        bin_mids = self.bin_mid_points
        bin_widths = self.bin_widths

        # pdf method
        if not self._freeze_temperature:

            def _maxwellian_pdf_optimisable(amplitude, temperature):
                # calculate the pdf
                pdf = _maxwellian_pdf(bin_mids, temperature) * bin_widths
                pdf_pul = spec2leth(pdf, self._group_structure)

                # return normalised and scaled pdf
                return amplitude * pdf / max(pdf_pul)

        else:

            def _maxwellian_pdf_optimisable(amplitude):
                # calculate the pdf
                pdf = _maxwellian_pdf(bin_mids, self._temperature) * bin_widths

                # return normalised and scaled pdf
                return amplitude * pdf / max(pdf)

        # store distribution
        return _maxwellian_pdf_optimisable

    @property
    def label(self):
        """Returns a string label describing the parameter values

        Returns
        -------
        label : str
            A string providing the distribution type and the parameter
            values of the distribution
        """
        return f"{self._type} (amp={self._amplitude:.2e} t={self._temperature} [{self._temperature_range[0]:.2e}, {self._temperature_range[1]:.2e}])"

    def __str__(self):
        """Returns a string summarising the parameters and their ranges"""
        string = (
            "maxwellianDistribution\n"
            f" - amplitude : {self._amplitude} [{self._amplitude_range[0]:.2e}, {self._amplitude_range[1]:.2e}]\n"
            f" - t (freeze = {self._freeze_temperature}) : {self._temperature:.2e} [{self._temperature_range[0]:.2}, {self._temperature_range[1]:.2e}]\n"
        )

        return string


class LogNormalDistribution(Distribution):
    """A log normal distribution"""

    def __init__(
        self,
        amplitude: float = 1.0,
        amplitude_range: list[float, float] = None,
        mode: float = 1.0,
        mode_range: list[float, float] = None,
        freeze_mode: bool = False,
        sigma: float = 1.0,
        sigma_range: list[float, float] = None,
        freeze_sigma: bool = False,
        interpolate_scale: str = "log",
    ):
        """A distribution providing a log normal pdf

        Parameters
        ----------
        amplitude : float, optional
            The amplitude applied to the normalised pdf,
            by default 1.0
        amplitude_range : list[float, float], optional
            The range of values of the amplitude applied to the normalsied
            pdf within which the optimistaion method can sample, by
            default [1e-2, 1e2]
        mode : float, optional
            The mode of the log normal distribution, by default 1.0
        mode_range : list[float, float], optional
            The range of the mode of the log normal distribution, between
            which the optimistation method can sample, by default [0.5, 1.5]
        freeze_mode : bool, optional
            Allows for the freezing of the mode, which will then remove this as
            a free parameter when passing the distribution to the optimistaion
            method, by default False
        sigma : float, optional
            The standard deviation of the log normal distribution, by default 1.0
        sigma_range : list[float, float], optional
            The range of the standard deviation, between which the optimistaion method
            can sample, defaults [0.5, 1.5]
        freeze_sigma : bool, optional
            Allows for the freezing of the standard deviation, which will then remove
            this as a free parameter when passing the distribution to the optimisation
            method, by default False
        interpolate_scale : str, optional
            The mid position to sample from within the goup structure, which
            can be either 'log' of 'lin' for logarithmic or linear space
            sampling respectively, by default "log"
        """

        # intialise parent
        super().__init__()

        # initialise parameters
        self._amplitude: float = amplitude
        self._amplitude_range: list[float, float] = (
            amplitude_range if amplitude_range else [1e-2, 1e2]
        )
        self._mode: float = mode
        self._mode_range: list[float, float] = mode_range if mode_range else [0.5, 1.5]
        self._freeze_mode: bool = freeze_mode
        self._sigma: float = sigma
        self._sigma_range: list[float, float] = (
            sigma_range if sigma_range else [0.5, 1.5]
        )
        self._freeze_sigma: bool = freeze_sigma
        self._interpolation_scale: str = interpolate_scale

        # set type
        self._type: str = "lognormal"

        # set initial parameters
        if self._freeze_mode and self._freeze_sigma:
            self._params = [self._amplitude]
            self._param_ranges = [self._amplitude_range]
        elif self._freeze_mode and not self._freeze_sigma:
            self._params = [self._amplitude, self._sigma]
            self._param_ranges = [self._amplitude_range, self._sigma_range]
        elif not self._freeze_mode and self._freeze_sigma:
            self._params = [self._amplitude, self._mode]
            self._param_ranges = [self._amplitude_range, self._mode_range]
        else:
            self._params = [self._amplitude, self._mode, self._sigma]
            self._param_ranges = [self._amplitude, self._mode_range, self._sigma_range]

    @property
    def params(self):
        """Returns the values of the free parameters

        Returns
        -------
        params : NDArray
            The free parameters of the distribution
        """
        return self._params

    @params.setter
    def params(self, params):
        """Sets the values of the free parameters

        Parameters
        ----------
        params : list[float, ...]
            Sets a list of the free paremters for the distribution
        """
        self._params = params

        if self._freeze_mode and self._freeze_sigma:
            self._amplitude = params[0]
        elif self._freeze_mode and not self._freeze_sigma:
            self._amplitude, self._sigma = params
        elif not self._freeze_mode and self._freeze_sigma:
            self._amplitude, self._mode = params
        else:
            self._amplitude, self._mode, self._sigma = params

    @property
    def distribution(self):
        """Returns a method providing the pdf of the SlopeDistribution which
        can be sampled with an array of energy values and a set of parameters

        returns
        -------
        distibution : Callable[NDArray, NDArray]
            A callable method which returns the pdf of the current distribution
            when provided with an array of energy positions to sample and the
            free parameters of the distribution

        """
        # get bin mid points and bin widths
        bin_mids = self.bin_mid_points
        # bin_widths = self.bin_widths

        # pdf methods
        if self._freeze_mode and self._freeze_sigma:

            def _log_normal_pdf_optimizable(amplitude):
                # calculate the pdf for the given group structure
                pdf_pul = _log_normal_pdf(bin_mids, self._mode, self._sigma)
                pdf_pul = amplitude * pdf_pul / max(pdf_pul)

                return leth2spec(pdf_pul, self.group_structure)

        elif self._freeze_mode and not self._freeze_sigma:

            def _log_normal_pdf_optimizable(amplitude, sigma):
                # calculate the pdf for the given group structure
                pdf_pul = _log_normal_pdf(bin_mids, self._mode, sigma)
                pdf_pul = amplitude * pdf_pul / max(pdf_pul)

                return leth2spec(pdf_pul, self.group_structure)

        elif not self._freeze_mode and self._freeze_sigma:

            def _log_normal_pdf_optimizable(amplitude, mode):
                # calculate the pdf for the given group structure

                pdf_pul = _log_normal_pdf(bin_mids, mode, self._sigma)
                pdf_pul = amplitude * pdf_pul / max(pdf_pul)

                return leth2spec(pdf_pul, self.group_structure)

        else:

            def _log_normal_pdf_optimizable(amplitude, mode, sigma):
                # calculate the pdf for the given group structure

                pdf_pul = _log_normal_pdf(bin_mids, mode, sigma)
                pdf_pul = amplitude * pdf_pul / max(pdf_pul)

                return leth2spec(pdf_pul, self.group_structure)

        return _log_normal_pdf_optimizable

    @property
    def label(self):
        """Returns a string label describing the parameter values

        Returns
        -------
        label : str
            A string providing the distribution type and the parameter
            values of the distribution
        """
        return f"{self._type} (amp={self._amplitude:.2e} mode={self._mode:.2e} sigma={self._sigma:.2e})"

    def __str__(self):
        """Returns a string summarising the parameters and their ranges"""
        string = (
            "LogNormalDistribution\n"
            f" - amplitude : {self._amplitude} [{self._amplitude_range[0]:.2e}, {self._amplitude_range[1]:.2e}]\n"
            f" - mode (freeze = {self._freeze_mode}) : {self._mode:.2e} [{self._mode_range[0]:.2e}, {self._mode_range[1]:.2e}]\n"
            f" - sigma (freeze = {self._freeze_sigma}) : {self._sigma:.2e} [{self._sigma_range[0]:.2e}, {self._sigma_range[1]:.2e}]\n"
        )

        return string


class GaussianDistribution(Distribution):
    """A gaussian distribution"""

    def __init__(
        self,
        amplitude: float = 1.0,
        amplitude_range: list[float, float] = None,
        mu: float = 1.0,
        mu_range: float = None,
        freeze_mu: bool = False,
        sigma: float = 1.0,
        sigma_range: float = None,
        freeze_sigma: bool = False,
        interpolation_scale: str = "log",
    ):
        """_summary_

        Parameters
        ----------
        amplitude : float, optional
            The amplitude applied to the normalised pdf,
            by default 1.0
        amplitude_range : list[float, float], optional
            The range of values of the amplitude applied to the normalsied
            pdf within which the optimistaion method can sample, by
            default [1e-2, 1e2]
        mu : float, optional
            The centroid of the gaussian distribution, by default 1.0
        mu_range : float, optional
            The range of the centroid of the gaussian, within which the
             optimistaion method can sample, by default [0.5, 1.5]
        freeze_mu : bool, optional
            _description_, by default False
        sigma : float, optional
            The standard deviation of the gaussian distribution, by default 1.0
        sigma_range : float, optional
            The range of the standard deviation, between which the optimistaion method
            can sample, defaults [0.5, 1.5]
        freeze_sigma : bool, optional
            Allows for the freezing of the standard deviation, which will then remove
            this as a free parameter when passing the distribution to the optimisation
            method, by default False
        interpolate_scale : str, optional
            The mid position to sample from within the goup structure, which
            can be either 'log' of 'lin' for logarithmic or linear space
            sampling respectively, by default "log"
        """

        # intialise parent
        super().__init__()

        # initial parameters
        self._amplitude: float = amplitude
        self._amplitude_range: list[float, float] = (
            amplitude_range if amplitude_range else [1e-2, 1e2]
        )
        self._mu: float = mu
        self._mu_range: float = mu_range if mu_range else [0.5, 1.5]
        self._freeze_mu: bool = freeze_mu
        self._sigma: float = sigma
        self._sigma_range: list[float, float] = (
            sigma_range if sigma_range else [0.5, 1.5]
        )
        self._freeze_sigma: bool = freeze_sigma
        self._interpolation_scale: str = interpolation_scale

        # set type
        self._type: str = "gaussian"

        # set initial parameters
        if self._freeze_mu and self._freeze_sigma:
            self._params = [self._amplitude]
            self._param_ranges = [self._amplitude_range]
        elif self._freeze_mu and not self._freeze_sigma:
            self._params = [self._amplitude, self._sigma]
            self._param_ranges = [self._amplitude_range, self._sigma_range]
        elif not self._freeze_mu and self._freeze_sigma:
            self._params = [self._amplitude, self._mu]
            self._param_ranges = [self._amplitude_range, self._mu_range]
        else:
            self._params = [self._amplitude, self._mu, self._sigma]
            self._param_ranges = [
                self._amplitude_range,
                self._mu_range,
                self._sigma_range,
            ]

    @property
    def params(self):
        """Returns the values of the free parameters

        Returns
        -------
        params : NDArray
            The free parameters of the distribution
        """
        return self._params

    @params.setter
    def params(self, params):
        """Sets the values of the free parameters

        Parameters
        ----------
        params : list[float, ...]
            Sets a list of the free paremters for the distribution
        """
        self._params = params

        if self._freeze_mu and self._freeze_sigma:
            self._amplitude = params[0]
        elif self._freeze_mu and not self._freeze_sigma:
            self._amplitude, self._sigma = params
        elif not self._freeze_mu and self._sigma:
            self._amplitude, self._mu = params
        else:
            self._amplitude, self._mu, self._sigma = params

    @property
    def distribution(self):
        """Returns a method providing the pdf of the SlopeDistribution which
        can be sampled with an array of energy values and a set of parameters

        returns
        -------
        distibution : Callable[NDArray, NDArray]
            A callable method which returns the pdf of the current distribution
            when provided with an array of energy positions to sample and the
            free parameters of the distribution

        """
        # get bin mid points and bin widths
        bin_mids = self.bin_mid_points
        bin_widths = self.bin_widths

        # pdf method
        if self._freeze_mu and self._freeze_sigma:

            def _gaussian_pdf_optimisable(amplitude):
                # calculate pdf and scale by bin widhts
                pdf = _gaussian_pdf(bin_mids, self._mu, self._sigma) * bin_widths
                pdf_pul = spec2leth(pdf, self._group_structure)

                # return normalised and scaled pdf
                return amplitude * pdf / max(pdf_pul)

        elif self._freeze_mu and not self._freeze_sigma:

            def _gaussian_pdf_optimisable(amplitude, sigma):
                # calculate pdf and scale by bin widhts
                pdf = _gaussian_pdf(bin_mids, self._mu, sigma) * bin_widths
                pdf_pul = spec2leth(pdf, self._group_structure)

                # return normalised and scaled pdf
                return amplitude * pdf / max(pdf_pul)

        elif not self._freeze_mu and self._freeze_sigma:

            def _gaussian_pdf_optimisable(amplitude, mu):
                # calculate pdf and scale by bin widhts
                pdf = _gaussian_pdf(bin_mids, mu, self._sigma) * bin_widths
                pdf_pul = spec2leth(pdf, self._group_structure)

                # return normalised and scaled pdf
                return amplitude * pdf / max(pdf_pul)

        else:

            def _gaussian_pdf_optimisable(amplitude, mu, sigma):
                # calculate pdf and scale by bin widhts
                pdf = _gaussian_pdf(bin_mids, mu, sigma) * bin_widths
                pdf_pul = spec2leth(pdf, self._group_structure)

                # return normalised and scaled pdf
                return amplitude * pdf / max(pdf_pul)

        return _gaussian_pdf_optimisable

    @property
    def label(self):
        """Returns a string label describing the parameter values

        Returns
        -------
        label : str
            A string providing the distribution type and the parameter
            values of the distribution
        """
        return f"{self._type} (amp={self._amplitude:.2e} mu={self._mu:.2e} sigma={self._sigma:.2e})"

    def __str__(self):
        """Returns a string summarising the parameters and their ranges"""
        string = (
            "GaussianDistribution\n"
            f" - amplitude : {self._amplitude:.2e} [{self._amplitude_range[0]:.2e}, {self._amplitude_range[1]:.2e}]\n"
            f" - mu (freeze = {self._freeze_mu}) : {self._mu:.2e} [{self._mu_range[0]:.2e}, {self._mu_range[1]:.2e}]\n"
            f" - sigma (freeze = {self._freeze_sigma}) : {self._sigma:.2e} [{self._sigma_range[0]:.2e}, {self._sigma_range[1]:.2e}]\n"
        )

        return string


class WattSpectrumDistribution(Distribution):
    """A Watt spectrum"""

    def __init__(
        self,
        amplitude: float = 1.0,
        amplitude_range: list[float, float] = None,
        a: float = 0.988,
        b: float = 2.249,
        interpolate_scale: str = "log",
    ):
        """A distribution providing a watt spectrum pdf

        Parameters
        ----------
        amplitude : float, optional
            The amplitude applied to the normalised pdf,
            by default 1.0
        amplitude_range : list[float, float], optional
            The range of values of the amplitude applied to the normalsied
            pdf within which the optimistaion method can sample, by
            default [1e-2, 1e2]
        a : float, optional
            Watt spectrum parameter a in MeV, by default 0.988
        b : float, optional
            Wall spectrum parameter b in MeV, by default 2.249
        interpolate_scale : str, optional
            The mid position to sample from within the goup structure, which
            can be either 'log' of 'lin' for logarithmic or linear space
            sampling respectively, by default "log"
        """
        # initialise parent
        super().__init__()

        # initial parameters
        self._amplitude: float = amplitude
        self._amplitude_range: list[float, float] = (
            amplitude_range if amplitude_range else [1e-2, 1e2]
        )
        self._a: float = a
        self._b: float = b
        self._interpolation_scale: str = interpolate_scale

        # set type
        self._type: str = "watt_spectrum"

        # set initiap parameters
        self._params: list[float] = [self._amplitude]
        self._param_ranges: list[float, float] = [self._amplitude_range]

    @property
    def params(self):
        """Returns the values of the free parameters

        Returns
        -------
        params : NDArray
            The free parameters of the distribution
        """
        return self._params

    @params.setter
    def params(self, params):
        """Sets the values of the free parameters

        Parameters
        ----------
        params : list[float, ...]
            Sets a list of the free paremters for the distribution
        """
        self._params = params
        self._amplitude = params[0]

    @property
    def distribution(self):
        """Returns a method providing the pdf of the SlopeDistribution which
        can be sampled with an array of energy values and a set of parameters

        returns
        -------
        distibution : Callable[NDArray, NDArray]
            A callable method which returns the pdf of the current distribution
            when provided with an array of energy positions to sample and the
            free parameters of the distribution

        """
        # get bin mid points and bin widths
        bin_mids = self.bin_mid_points
        bin_widths = self.bin_widths

        # pdf method
        def _watt_spectrum_optimisable(amplitude):
            # calculate the pdf
            pdf = _watt_spectrum_pdf(bin_mids, self._a, self._b) * bin_widths
            pdf_pul = spec2leth(pdf, self._group_structure)

            # return normalised and scaled pdf
            return amplitude * pdf / max(pdf_pul)

        return _watt_spectrum_optimisable

    @property
    def label(self):
        """Returns a string label describing the parameter values

        Returns
        -------
        label : str
            A string providing the distribution type and the parameter
            values of the distribution
        """
        return (
            f"{self._type} (amp={self._amplitude:.2e} a={self._a:.2}, b={self._b:.2e})"
        )

    def __str__(self):
        """Returns a string summarising the parameters and their ranges"""
        string = (
            "WattSpectrumDistribution\n"
            f" - amplitude : {self._amplitude} [{self._amplitude_range[0]:.2e}, {self._amplitude_range[1]:.2e}]\n"
            f" - a : {self._a:.2e}\n"
            f" - b : {self._b:.2e}\n"
        )

        return string


class ParametricSpectrum(DataHandler):
    """parametrixSpectrum which contains the a priori set of distributions and
    an optimisation method to provide a fitted set of solution distributions
    """

    def __init__(self, distributions_a_priori: list[Distribution], *args, **kwargs):
        """parametrixSpectrum which contains the a priori set of distributions and
        an optimisation method to provide a fitted set of solution distributions

        Parameters
        ----------
        distributions_a_priori : list[Distribution]
            A list of distributions to combine into a parametric spectrum
        """
        super().__init__(*args, **kwargs)

        # set up store of distributions
        self._distributions_a_priori: list[Distribution] = distributions_a_priori
        self._distributions_solutions: list[Distribution] = []

    @property
    def distributions_a_priori(self):
        """Returns a list of the a priori distributions

        Returns
        -------
        a_priori_distributions : list[Distribution]
            A list of the a priori distributions
        """
        return self._distributions_a_priori

    @property
    def distributions_solutions(self):
        """Returns a list of the a solution distributions

        Returns
        -------
        solution_distributions : list[Distribution]
            A list of the solution distributions
        """
        return self._distributions_solutions

    @property
    def a_priori(self):
        """Gets the a_priori attribute

        Returns
        -------
        a_priori : Union[list[float], np.typing.NDArray]
            A priori spectrum
        """

        self._a_priori = np.zeros(self._n)

        for distribution_a_priori in self._distributions_a_priori:
            self._a_priori += distribution_a_priori.sample

        return self._a_priori

    def scale_a_priori(self):
        """Will scale the a priori by matching the measured values and the
        folded a priori and response matrix

        Returns
        -------
        scale_factor : float
            The factor which the a priori spectrum has been  scaled
        """
        # Calculate reaction rates from a priori spectrum
        unfitted = np.dot(self._response_matrix, self._a_priori)

        # calculate scale factor
        scale_factor = sum(unfitted) / sum(unfitted**2 / self._measured)

        # set a priori
        self._a_priori *= scale_factor

        return scale_factor

    def set(self, *args, **kwargs):
        """sets the various vectors and matrices required

        Parameters
        ----------
        label : str
            Label of the vector or matrix requiring to be set, such as
            a_priori, measured or response matrix.
        data : Union[list[float], np.typing.NDArray]
            Data associlated with label to be set.
        uncertainty : Union[list[float], np.typing.NDArray], optional
            Uncertainty associlated with the label to be set
        covariance : Union[list[float], np.typing.NDArray], optional
            Covariance associated with the label to be set
        """
        super().set(*args, **kwargs)

        for distribution in self._distributions_a_priori:
            distribution.set(*args, **kwargs)

    def set_all(self, *args, **kwargs):
        """sets all of the vectors and matrices from a DataHandler or child of
        a DataHandler

        Parameters
        ----------
        data_handler : DataHandler
            A data hanlder or child of, from which any settable data
            is transfered from this to the instance of the class
            with settable labels defined in self._settable_attributes
        """
        super().set_all(*args, **kwargs)

        for distribution in self._distributions_a_priori:
            distribution.set_all(*args, **kwargs)

    def run(self, minimize_tol=0.01):
        """Tun method to call the optimisatio nmethod and produce a set of
        gitted parameters

        Parameters
        ----------
        minimize_tol : float, optional
            The tol parameter passed to the scipy minimize optimisation method,
            by default 0.01
        """
        # collect the initial parameters for all distributions
        param_indices = (
            []
        )  # stores the position of the parameters for each distribution
        params_initial = []  # stores the initial parameters
        param_ranges = []  # stores the bounds for the parameters

        # set an empty array to store summed distributions of a priori
        self._a_priori = np.zeros(self._n)  # set up empty a priori spectrum

        # scale the ampltiude of each distribution and collect the initial parameters
        for dist_a_priori in self._distributions_a_priori:
            self._a_priori += dist_a_priori.sample

        # scale a priori to fit calculated against measured values
        scale_factor = self.scale_a_priori()

        # copy and scale the a priori to allow optimiser to change solution distribution parameters
        self._distributions_solutions = [
            copy.deepcopy(d) for d in self._distributions_a_priori
        ]

        # scale the proposed solution distributions
        for dist_solution in self._distributions_solutions:
            # scale the amplitude of each distribution
            dist_solution.scale_amplitude(scale_factor)

            # collect the parameters and flatten into single array
            # for passing to optimisation method
            param_indices.append(
                [len(params_initial), len(params_initial) + len(dist_solution.params)]
            )
            params_initial.extend(dist_solution.params)
            param_ranges.extend(dist_solution._param_ranges)

        # set up fitness function for optimisation
        def fitness(params):
            """Fitness function which accepts the free parmeters for
            all distributions and returns a chi squared per degree of
            freedom between the measured and calculated values

            Parameters
            ----------
            params : NDArray
                An array of the free paremeters to trial

            Returns
            -------
            chi_sq_per_degree_of_freedom : float
                The chi sq per degree of freedom for the measured and
                calculated values from the current parameters trialled
            """
            # set up empty np array
            spectrum = np.zeros(self._distributions_a_priori[0].shape_n)

            # add sampled distributions to solution spectrum
            for dist_solution_trial, (i_min, i_max) in zip(
                self._distributions_solutions, param_indices
            ):
                # create new instance of distribution with new
                # parameters to trial and group structure
                dist_solution_trial.params = params[i_min:i_max]

                # add the distribution to the total spectrum
                spectrum += dist_solution_trial.sample

            # calculate equivalent of measured for current solution, scaling by amplitude parameter
            calculated = np.dot(self._response_matrix, spectrum)

            # calculate the chi squared per degree of freedom
            chi_sq = (
                np.sum(
                    ((self._measured - calculated) ** 2)
                    / (self._measured_uncertainty**2)
                )
                / self._m
            )

            return chi_sq

        # optimise parameters
        # optimisation_solution = minimize(
        #     fitness, x0=params_initial, bounds=param_ranges, method="L-BFGS-B"
        # )

        optimisation_solution = basinhopping(
            fitness,
            params_initial,
            minimizer_kwargs={"method": "L-BFGS-B", "bounds": param_ranges},
        )

        # set up solution spectrum
        self._solution = np.zeros(self._distributions_a_priori[0].shape_n)

        # collect and sum solution distributions
        for dist_solution, (i_min, i_max) in zip(
            self._distributions_solutions, param_indices
        ):
            dist_solution.params = optimisation_solution.x[i_min:i_max]
            self._solution += dist_solution.sample

    def __add__(self, other: Union[Distribution, "ParametricSpectrum"]):
        """Addition of the distribution with another distribution or
        a parametric spectrum, producing a parametric spectrum.

        Parameters
        ----------
        other : Union[ParametricSpectrumt, Distribution]
            Either a ParametricSpectrum instance or a Distribution (or child)
            instance to be summed in order to produce a ParametricSpectrum with
            multiple distributions.

        Returns
        -------
        parametric_spectrum : ParametricSpectrum
            A parametric spectrum prduced by the summation of multiple distributions

        Raises
        ------
        Exception
            A type exception is raised if the other instance is not a ParametricSpectrum
            or distriburtion (or child) instance
        """
        if isinstance(other, ParametricSpectrum):
            return ParametricSpectrum(
                [*self._distributions_a_priori, *other.distributions_a_priori]
            )
        elif isinstance(other, Distribution):
            return ParametricSpectrum([*self._distributions_a_priori, other])
        else:
            msg = (
                "[ERROR] - only Paramaters and ParametricSpectra "
                f"can be combined not {type(other)}"
            )
            raise TypeError(msg)


def set_of_gaussian_distributions(
    e_min: float,
    e_max: float,
    amp_min: float,
    amp_max: float,
    sigma: float,
    n_gaussians: int,
    e_space: str = "log",
    amp_space: str = "log",
):
    """Returns a set of gaussian distributions

    Parameters
    ----------
    e_min : float
        The centroid of the gaussian with the lowest energy
    e_max : float
        The centroid of the gaussian with the highest energy
    amp_min : float
        The amplitude of the gaussian with the lowest energy
    amp_max : float
        The ampltuide of the gaussian with the highest energy
    sigma : float
        The standard detiavtion of each gaussian
    n_gaussians : int
        The number of gaussian distributions to generate and return
    e_space : str, optional
        The centroid energies can either be space logarithmically or
        linearly beither either passing 'log' or 'lin, by default "log"
    amp_space : str, optional
        The amplitude spacings can either be space logarithmically or
        linearly beither either passing 'log' or 'lin, by default "log"

    Returns
    -------
    distributions : list[Distribution]
        A list of the gaussian distributions generated
    """
    if e_space == "lin":
        energies = np.linspace(e_min, e_max, n_gaussians)
    elif e_space == "log":
        energies = np.logspace(np.log10(e_min), np.log10(e_max), n_gaussians)
    else:
        msg = f"e_space shoudl be either lin or log not {e_space}"
        raise ValueError(msg)

    if amp_space == "lin":
        amplitudes = np.linspace(amp_min, amp_max, n_gaussians)
    elif amp_space == "log":
        amplitudes = np.logspace(np.log10(amp_min), np.log10(amp_max), n_gaussians)
    else:
        msg = f"amp_space shoudl be either lin or log not {amp_space}"
        raise ValueError(msg)

    gaussians = []

    for energy, amplitude in zip(energies, amplitudes):
        gaussian = GaussianDistribution(amplitude=amplitude, mu=energy, sigma=sigma)
        gaussians.append(gaussian)

    return gaussians


def set_of_fixed_lognormal_distributions(
    e_min: float = 1e0,
    e_max: float = 1e0,
    amp_min: float = 1e0,
    amp_max: float = 1e0,
    sigma: float = 1.0,
    n_lognorms: int = 1,
    e_space: str = "log",
    amp_space: str = "log",
):
    """Returns a set of log normal distributions

    Parameters
    ----------
    e_min : float, optional
        The mode of the log normal with the lowest energy, by default 1e0
    e_max : float, optional
        The mode of the log normal with the highest energy, by default 1e0
    amp_min : float, optional
        The amplitude of the log normal with the lowest energy, by default 1e0
    amp_max : float, optional
        The amplitude of the log normal with the highest energy, by default 1e0
    sigma : float, optional
        The standard deviation of the log normals generated, by default 1.0
    n_lognorms : int, optional
        The number of log normal distributions to generate, by default 1
    e_space : str, optional
        The centroid energies can either be space logarithmically or
        linearly beither either passing 'log' or 'lin, by default "log"
    amp_space : str, optional
        The amplitude spacings can either be space logarithmically or
        linearly beither either passing 'log' or 'lin, by default "log"

    Returns
    -------
    distributions list[Distribution]
        A list of the log normal distributions generated
    """

    if e_space == "lin":
        energies = np.linspace(e_min, e_max, n_lognorms)
    elif e_space == "log":
        energies = np.logspace(np.log10(e_min), np.log10(e_max), n_lognorms)
    else:
        msg = f"e_space should be either lin or log not {e_space}"
        raise ValueError(msg)

    if amp_space == "lin":
        amplitudes = np.linspace(amp_min, amp_max, n_lognorms)
    elif amp_space == "log":
        amplitudes = np.logspace(np.log10(amp_min), np.log10(amp_max), n_lognorms)
    else:
        msg = f"amp_space should be either lin or log not {amp_space}"
        raise ValueError(msg)

    # set up list to store log normal distributions
    lognorms = []

    # iterate over energy and cmpllitude ranges
    for energy, amplitude in zip(energies, amplitudes):
        lognorm = LogNormalDistribution(
            amplitude=amplitude,
            amplitude_range=[amplitude * 1e-3, amplitude * 1e3],
            mode=energy,
            sigma=sigma,
            freeze_mode=True,
            freeze_sigma=True,
        )
        lognorms.append(lognorm)

    return lognorms
