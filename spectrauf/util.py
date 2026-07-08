"""Util contains a range of utilities for pre and post processing of spectra, along with
anscilliary methods for mathematics and data handling.
"""

from typing import Union
from dataclasses import dataclass
from math import log10
import warnings
import numpy as np
from numpy.typing import NDArray

from .checkvalues import is_monotonically_ascending
from .logger import get_logger


def rebin(
    histogram: Union[list[float], NDArray],
    bounds: Union[list[float], NDArray],
    bounds_rebin: Union[list[float], NDArray],
    method: str = "linear",
):
    """Rebinning method to re-bin between two group structures

    Parameters
    ----------
    histogram : Union[list[float], NDArray]
        A list or array representing the values in a histogram which
        is to be re-binned
    bounds : Union[list[float], NDArray]
        The original group structure of the histogram
    bounds_rebin : Union[list[float], NDArray]
        The new group structure into which the histogram is to be re-binned
    method : str, optional
        The space across which the re-binning is applied, which can either
        be 'linear' or 'log', by default "linear"

    Returns
    -------
    histogram_rebinned : NDArray
        The re-binned histogram
    """

    logger = get_logger()
    range_old = f'energy range of histogram = {bounds[0]:.2e} - {bounds[-1]:.2e}'
    range_new = f'energy range of new structure = {bounds_rebin[0]:.2e} - {bounds_rebin[-1]:.2e}'
    logger.debug(
        f"running rebin - {range_old} - {range_new}"
    )

    if method not in ["linear", "log"]:
        msg = (
            f"Method key word argument should either be linear or log:"
            f"{method} provided"
        )
        raise ValueError(msg)

    # Error checking. New bins need to fall within bounds of old bins
    if bounds_rebin[0] < bounds[0]:
        logger.info("New bins fall bellow lower bounds of old")
    elif bounds_rebin[-1] > bounds[-1]:
        logger.info("New bins fall above upper bounds of old")

    # copy the original hist
    hist = np.array(histogram, np.float64)
    hist_original = hist.copy()
    bounds = np.array(bounds, np.float64)

    # set up array to store rebinned histogram
    hist_rebinned = np.zeros(len(bounds_rebin) - 1, dtype=np.float64)

    # check bounds and bounds_rebin are monotonically ascending
    if not is_monotonically_ascending(bounds):
        msg = "[ERROR] - rebin received bounds which are not monotonically ascending"
        raise ValueError(msg)

    if not is_monotonically_ascending(bounds_rebin):
        msg = "[ERROR] - rebin received bounds_rebin which are not monotonically ascending"
        raise ValueError(msg)

    for i_rebin, (bound_rebin_lo, bound_rebin_up) in enumerate(
        zip(bounds_rebin[:-1], bounds_rebin[1:])
    ):
        for i, (bound_lo, bound_up) in enumerate(zip(bounds[:-1], bounds[1:])):
            # Logic 1: If new bin encompass old bin
            #      old bin        |-------------|
            #      new bin     |--------------------|
            if bound_rebin_lo <= bound_lo and bound_rebin_up >= bound_up:
                hist_rebinned[i_rebin] += hist[i]

            # Logic 2: If new bin falls within old bin
            #      old bin    |-------------------|
            #      new bin        |-----------|
            elif bound_rebin_lo > bound_lo and bound_rebin_up < bound_up:
                if method == "linear":
                    fraction_of_old = (bound_rebin_up - bound_rebin_lo) / (
                        bound_up - bound_lo
                    )
                else:
                    fraction_of_old = (
                        log10(bound_rebin_up) - log10(bound_rebin_lo)
                    ) / (log10(bound_up) - log10(bound_lo))

                # add fraction of host value to rebinned histogram
                hist_rebinned[i_rebin] += fraction_of_old * hist[i]

            # Logic 3: If new bin cover lower boundary
            #      old bin             |---------------|
            #      new bin     |-------------|
            elif (bound_rebin_lo <= bound_lo) and (
                bound_lo < bound_rebin_up < bound_up
            ):
                if method == "linear":
                    fraction_of_old = (bound_rebin_up - bound_lo) / (
                        bound_up - bound_lo
                    )
                else:
                    fraction_of_old = (log10(bound_rebin_up) - log10(bound_lo)) / (
                        log10(bound_up) - log10(bound_lo)
                    )

                # add fraction of host value to rebinned histogram
                hist_rebinned[i_rebin] += fraction_of_old * hist[i]

            # Logic 4: If new bin cover upper boundary
            #      old bin     |---------------|
            #      new bin               |-------------|
            elif bound_lo < bound_rebin_lo < bound_up and bound_rebin_up >= bound_up:
                if method == "linear":
                    fraction_of_old = (bound_up - bound_rebin_lo) / (
                        bound_up - bound_lo
                    )
                else:
                    fraction_of_old = (log10(bound_up) - log10(bound_rebin_lo)) / (
                        log10(bound_up) - log10(bound_lo)
                    )

                # add fraction of host value to rebinned histogram
                hist_rebinned[i_rebin] += fraction_of_old * hist[i]

        # prune values and bounds from hist and bounds once they have been assigned
        while len(bounds) > 1 and bound_rebin_lo >= bounds[1]:
            hist = hist[1:]
            bounds = bounds[1:]

    # check for conservation of values
    if (bounds_rebin[0] <= bounds[0]) and (bounds_rebin[-1] >= bounds[-1]):
        sum_hist = sum(hist_original)
        sum_hist_rebinned = sum(hist_rebinned)
        tol = 1e-4
        if not (
            (sum_hist * (1 - tol)) <= sum_hist_rebinned <= ((sum_hist * (1 + tol)))
        ):
            msg = (
                "rebinned histogram has not conserved total value, "
                f"sum(hist)={sum_hist:.8e} sum(hist_rebinned = {sum_hist_rebinned:.8e})"
            )
            logger.fata(msg)
            raise ValueError(msg)
        else:
            logger.info(
                f"rebinned histogram has conserved total value sum(hist)={sum_hist:.8e} sum(hist_rebinned = {sum_hist_rebinned:.8e})"
            )

    return hist_rebinned


def convert_spectrum_to_lethargy(
    flux: np.typing.NDArray, group_structure: np.typing.NDArray
):
    """Converts a spectrum from flux to flux per unit lethargy using the
    squation phi / ln(e_upper/e_lower)

    Parameters
    ----------
    spectrum : iter
        The spectrum to be converted to flux per unit lethargy

    group_structure : iter
        The group structure of the spectrum
    """

    # ensure flux and group structure are numpy arrays
    flux = np.array(flux)
    group_structure = np.array(group_structure)

    # calculate flux per unit lethargy
    flux_per_unit_lethargy = flux / np.log(group_structure[1:] / group_structure[:-1])

    # return flux per unit lethargy
    return flux_per_unit_lethargy


def convert_lethargy_to_spectrum(
    flux_per_unit_lethargy: np.typing.NDArray, group_structure
):
    """Converts a spectrum from flux to flux per unit lethargy using the
    squation lethargy * ln(e_upper/e_lower)

    Parameters
    ----------
    lethargy : iter
        The spectrum in flux per unit lethargy to be converted to flux per unit
         lethargy

    group_structure : iter
        The group structure of the spectrum
    """

    # ensure flux per unit lethargy and group structure are numpy arrays
    flux_per_unit_lethargy = np.array(flux_per_unit_lethargy)
    group_structure = np.array(group_structure)

    # calculate flux per unit lethargy
    flux_per_unit_lethargy = flux_per_unit_lethargy * np.log(
        group_structure[1:] / group_structure[:-1]
    )

    # return flux per unit lethargy
    return flux_per_unit_lethargy


def kullback_leibler_divergence(vec_p, vec_q):
    """Calculates the Kullback-Leibler divergence of two vectors (spectra)

    Parameters
    ----------
    spectrum_1 : iter
        A vector, such as a spectrum.
    spectrum_2 : iter
        A vector, such as a spectrum

    Returns
    -------
    kullback_leibler_divergence : float
        The Kullback-Leibler divergence
    """
    if (
        isinstance(vec_p, np.ndarray)
        and isinstance(vec_q, np.ndarray)
        and vec_p.shape == vec_q.shape
    ):
        pass
    elif (
        isinstance(vec_p, (list, tuple))
        and isinstance(vec_q, (list, tuple))
        and len(vec_q) == len(vec_q)
    ):
        vec_p = np.array(vec_p)
        vec_q = np.array(vec_q)
    else:
        msg = "kullback_leibler_divergence requires 1d iterable or numpy array"
        raise TypeError(msg)

    return sum(vec_p * np.log(vec_p / vec_q))


def read_fispact_reference_spectrum(input_filepath: str):
    """
    method to read in fispact flux input files and return numpy arrays with their
    group structure and flux

    Parameters
    ----------
    input_filepath : str
        The input file path of the fispact flux file

    Returns
    -------
    group_structure : np.typing.NDArray
        The energy bin bounds in MeV in ascending order

    flux : np.typing.NDArray
        The flux spectrum in cm^-2
    """

    # get logger
    logger = get_logger()

    with open(input_filepath, "r", encoding="UTF8") as input_file:
        rows = [r.replace(r"\n", "") for r in input_file]

    # count number of energy bins
    n_bins = int((len(rows) - 4) / 2)

    logger.info(rows[: n_bins + 1])

    # group_structure and flux
    group_structure = np.array(rows[: n_bins + 1], dtype=np.float64)[::-1] / 1e6
    flux = np.array(rows[n_bins + 2 : -2], dtype=np.float64)[::-1]

    # return group_structure and flux
    return group_structure, flux


@dataclass
class FispactReferenceSpectrum:
    """FispactReferenceSpectrum stores the data from the
    FISPACT-II reference spectra
    """

    group_structure: Union[list[float], np.typing.NDArray]
    flux: Union[list[float], np.typing.NDArray]
    label: str

    @property
    def n_bins(self):
        """returns the number of bins in the group structure

        Returns
        -------
        number_of_bins : int
            The number of bins in the group structure
        """
        return len(self.flux)

    @property
    def flux_per_unit_lethargy(self):
        """The flux in per unit lethargy

        Returns
        -------
        flux_per_unit_lethargy : NDArray
            The flux in per unit lethargy
        """
        flux_pul = convert_spectrum_to_lethargy(self.flux, self.group_structure)
        return flux_pul


def kelvin_to_mev(temperature_in_kelvin):
    """Converts energy of a particle in kelvin to MeV

    Parameters
    ----------
    temperature_in_kelvin : _type_
        _description_

    Returns
    -------
    energy : float
        The energy of particles in MeV
    """
    boltzmann = 1.380649e-23  # J k^-1
    joule_per_mev = 1.602176634e-13  # C
    return temperature_in_kelvin * boltzmann / joule_per_mev


def centigrade_to_mev(temperature_in_centrigrade):
    """Converts energy of a particle from centrigrade to MeV

    Parameters
    ----------
    temperature_in_centrigrade : float
        The tmperature to convert

    Returns
    -------
    energy : float
        The energy of particles in MeV
    """
    temperature_in_kelvin = temperature_in_centrigrade + 273.15
    return kelvin_to_mev(temperature_in_kelvin)


def strip_leading_or_trailing_zeros(vector: Union[list[float], NDArray]):
    """Strips the leading or training zeros from a vector, and returns
    the stripped vector and the index range of the returned section of the
    vector

    Parameters
    ----------
    vector : Union[list[float], NDArray]

    """

    # ensure vector is a numpy array
    vector = np.asarray(vector)

    # find index of all non-zero values
    nonzero = np.nonzero(vector)[0]

    # check for an array with only zeros
    if len(nonzero) == 0:
        return np.array([]), 0, 0

    # find the index of the first and last non-zero value
    start = nonzero[0]
    end = nonzero[-1] + 1

    # return the trimmed vector along with the index positions
    trimmed = vector[start:end]
    return trimmed, start, end


def reduced_chi_squared(
    vec_model: Union[list[float], NDArray],
    vec_measured: Union[list[float], NDArray],
    vec_measured_abs_err: Union[list[float], NDArray],
):
    """Returns the reduced chi squared of the solution measured values
    assiciated with the solution

    Returns
    -------
    reduced_chi_squared : float
        Reduced chi squared of the solution measured values
        assiciated with the solution
    """
    # calculate the chi-squared of the measured and fitted values
    chi_sq = ((vec_model - vec_measured) / vec_measured_abs_err) ** 2

    # return the reduced chi squared
    return sum(chi_sq) / len(vec_model)
