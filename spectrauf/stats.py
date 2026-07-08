"""Stats provides functions and classes that provide quantification of
various statistical properties for a given unfolding problem
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def dot_product_angle(
    vector_1: Union[list[float], NDArray], vector_2: Union[list[float], NDArray]
):
    """Calculates the angle between two vectors

    parameters
    ----------
    vector_1 : Union[list[float], NDArray]
        The first vector of length m
    vector_2 : Union[list[float], NDArray]
        The second vector of length m

    returns
    -------
    angle : float
        The angle between vector_1 and vector_2

    """
    if np.linalg.norm(vector_1) == 0 and np.linalg.norm(vector_2):
        msg = "A vector has no magnitude"
        raise ValueError(msg)

    dot_product = np.dot(vector_1, vector_2)
    v1_norm = np.linalg.norm(vector_1)
    v2_norm = np.linalg.norm(vector_2)
    arccos = np.arccos(dot_product / (v1_norm * v2_norm))
    return np.degrees(arccos)


def _plot_matrix(matrix: NDArray, title, fig_size=(12, 8), tick_step=1):
    """Plots the values within a matrix using a heatmap

    parameters
    ----------
    matrix : NDArray
        A matrix
    """

    plt.rcParams["font.size"] = 30
    fig, axis = plt.subplots(1, figsize=fig_size)
    pos = axis.imshow(matrix)
    fig.colorbar(pos, ax=axis)
    axis.set_title(title)
    tick_min = matrix.shape[0] - 1
    tick_max = matrix.shape[0]
    ticks = np.linspace(0, tick_min, tick_max)[::tick_step]
    labels = np.linspace(1, matrix.shape[0], matrix.shape[0])[::tick_step]

    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)

    plt.show()


class StatsResponseMatrix:
    """StatsResponseMatrix is a class that provides various metrics for
    the condition of a given matrix.
    """

    def __init__(self, response_matrix):
        """Initialise_class"""
        self._rm = response_matrix
        self._row_angles = np.zeros((self._rm.shape[0], self._rm.shape[0]))
        self._col_angles = np.zeros((self._rm.shape[1], self._rm.shape[1]))

        self._calculate_angles_of_vectors()

    @property
    def response_matrix(self):
        """Returns the response matrix"""
        return self._rm

    @property
    def m(self):
        """returns the number of rows in the response matrix"""
        return self._rm.shape[0]

    @property
    def n(self):
        """Returns the number of columns in the response matrix"""
        return self._rm.shape[1]

    def _calculate_angles_of_vectors(self):
        """Returns the angles between the row and column vectors"""

        # calculate angle between each row
        for i, row_i in enumerate(self._rm):
            for j, row_j in enumerate(self._rm):
                angle = dot_product_angle(row_i, row_j)
                if isinstance(angle, type(np.NaN)):
                    self._row_angles[i][j] = angle
                else:
                    self._row_angles[i][j] = 0

        for i, col_i in enumerate(self._rm.T):
            for j, col_j in enumerate(self._rm.T):
                angle = dot_product_angle(col_i, col_j)
                if isinstance(angle, type(np.NaN)):
                    self._col_angles[i][j] = angle
                else:
                    self._col_angles[i][j] = 0

    def plot_rows(self, fig_size=(12, 8), tick_step=1):
        """Plots the angle between the row vectors in the response matrix"""
        _plot_matrix(self._row_angles, "Rows", fig_size=fig_size, tick_step=tick_step)

    def plot_cols(self, fig_size=(12, 8), tick_step=1):
        """Plots the angle between the column vectors in the response matrix"""
        _plot_matrix(
            self._col_angles, "Columns", fig_size=fig_size, tick_step=tick_step
        )
