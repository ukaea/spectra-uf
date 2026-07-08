"""Serialisation methods to control the format of data loaded and saved to files"""

from typing import Union
from os import PathLike
import numpy as np
from numpy.typing import NDArray, DTypeLike


class Serialise:
    """Serialisation method to control the format from and to which data is
    loaded or saved.
    """


class SerialiseCSV(Serialise):
    """Comma seperated value (csv) Serialisation for the loading and saving of
    vectors and matrices in csv format
    """

    @staticmethod
    def load(input_path: Union[str, PathLike], np_dtype: DTypeLike):
        """Load method will load values from a comma seperated value (csv)
        file and return them as a numpy array

        Parameters
        ----------
        input_path : str
            Input file path for the comma seperated value (csv) file
        nd_dtype : DTypeLike
            Data type of numpy array returned
        """
        return np.loadtxt(input_path, dtype=np_dtype, delimiter=",")

    @staticmethod
    def save(output_path: Union[str, PathLike], data: NDArray):
        """Save method will save values from a numpy array to a comma
        seperated value (csv) file

        Parameters
        ----------
        output_path : str
            Output file path for the comma seperated value (csv) file
        data : NDArray
            Data to be saved to a comma sperated value (csv) file
        """
        np.savetxt(output_path, data, delimiter=",")


class SerialiseSSV:
    """Space seperated value (ssv) Serialisation for the loading and saving of
    vectors and matrices in ssv format
    """

    @staticmethod
    def load(input_path: Union[str, PathLike], np_dtype: DTypeLike):
        """Load method will load values from a space seperated value (ssv)
        file and return them as a numpy array

        Parameters
        ----------
        input_path : str
            Input file path for the space seperate value (ssv) file
        nd_dtype : DTypeLike
            Data type of numpy array returned
        """
        return np.loadtxt(input_path, dtype=np_dtype, delimiter=" ")

    @staticmethod
    def save(output_path: Union[str, PathLike], data: NDArray):
        """Save method will save values from a numpy array to a space
        seperated value (ssv) file

        Parameters
        ----------
        output_path : str
            Output file path for the space seperate value (ssv) file
        data : NDArray
            Data to be saved to a space seperate value (ssv) file
        """
        np.savetxt(output_path, data, delimiter=" ")
