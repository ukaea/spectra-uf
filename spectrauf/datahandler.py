"""Data handler is the base class for unfolding methers and utilities, with
common set and load methods, and a solution class returned after each unfolding
method is run.
"""

import numpy as np
from typing import Union
from os import PathLike

from .serialise import Serialise, SerialiseCSV
from .logger import get_logger
from .uncertanties import vector_uncertainty_to_cov as vuc
from .uncertanties import vector_cov_to_uncertainty as vcu
from .uncertanties import matrix_uncertainty_to_cov as muc
from .uncertanties import matrix_cov_to_uncertainty as mcu
from .checkvalues import (
    check_vals,
    is_numeric,
    is_integer,
    is_float,
    is_string,
    is_pathlike,
    is_iterable_numeric,
    is_iterable_integer,
    is_iterable_float,
    is_iterable_string,
    is_iterable_pathlike,
    val_passed_is_numeric,
    val_passed_is_integer,
    val_passed_is_float,
    val_passed_is_string,
    val_passed_is_pathlike,
    iterable_passed_is_numeric,
    iterable_passed_is_integer,
    iterable_passed_is_float,
    iterable_passed_is_string,
    iterable_passed_is_pathlike,
)


class DataHandler:
    """Data storage and interface class that acts as the base class for
    unfolding methods

    Parameters
    ----------
    serialiser : Serialise, optional
        Class containing methods that defines the load and save formats

    np_type : DTypeLike, optional
        Sets the required numpy data type stored in each ndarray

    Attributes
    ----------
    a_priori : NDArray
        A priori spectrum

    a_priori_uncertainty : NDArray
        a priori spectrum absolute uncertainty

    a_priori_covariance : NDArray
        a priori covariance matrix

    response_matrix : NDArray
        reponse matrix

    response_matrix_uncertainty : NDArray
        reponse matrix absolute uncertainty

    reponse_matrix_covariance : NDArray
        reponse matrix coraviance matrix

    measured : NDArray
        measured values

    measured_uncertainty : NDArray
        measured absolute uncertainty uncertainty

    measured_covariance : NDArray
        measured covariance matrix

    group structure : NDArray
        energy group structure

    target : NDArray
        a target spectrum if solution is already known

    solution : NDArray
        solution spectrum

    shape_m : int
        number of rows in reponse matrix and number of measured values

    shape_n : int
        number of columns in response matrix and number of bin in spectra
    """

    def __init__(
        self,
        serialiser: Serialise = SerialiseCSV,
        np_dtype: np.typing.DTypeLike = np.float64,
    ):
        """Intialise class"""
        self.logger = get_logger()

        # serialise inpout and output of numpy objects
        self._serialiser: Serialise = serialiser

        # paramters
        self._np_dtype: np.typing.DTypeLike = np_dtype

        # unfolding data
        self._a_priori = None
        self._a_priori_uncertainty = None
        self._a_priori_covariance = None

        self._response_matrix = None
        self._response_matrix_uncertainty = None
        self._response_matrix_covariance = None

        self._measured = None
        self._measured_uncertainty = None
        self._measured_covariance = None

        self._group_structure = None

        self._target = None
        self._target_uncertainty = None
        self._target_covariance = None

        self._solution = None

        # dimensions of vectors and matrices
        self._m = None
        self._n = None

        # run methods, to be overwritten in each unfolding method
        self._methods: dict = {}

        # name of the unfolding method
        self._method: str = "DataHandler"

        # lists of settable and savable attributes
        self._settable_attributes: list[str] = [
            "a_priori",
            "response_matrix",
            "measured",
            "group_structure",
            "target",
        ]

        self._savable_attributes: list[str] = [
            "a_priori",
            "response_matrix",
            "measured",
            "group_structure",
            "target",
            "solution",
        ]

    # set, load and save
    @check_vals(
        variable_type={
            "label": is_string,
            "data": is_iterable_numeric,
            "uncertainty": is_iterable_numeric,
            "covariance": is_iterable_numeric,
        }
    )
    def set(
        self,
        label: str,
        data: Union[list[float], np.typing.NDArray],
        uncertainty: Union[list[float], np.typing.NDArray] = None,
        covariance: Union[list[float], np.typing.NDArray] = None,
    ):
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

        self.logger.info(f"[SETTING]: {label}")

        if label not in self._settable_attributes:
            avail: str = "\n    ".join(self._settable_attributes)
            msg: str = (
                f"unable to use general set method for attribute {label} "
                f"available options are {avail}"
            )
            raise KeyError(msg)

        # dynamically set the attribute
        setattr(self, label, data)

        if not isinstance(uncertainty, type(None)) and isinstance(
            covariance, type(None)
        ):
            setattr(self, f"{label}_uncertainty", uncertainty)
        elif not isinstance(covariance, type(None)) and isinstance(
            uncertainty, type(None)
        ):
            setattr(self, f"{label}_covariance", covariance)
        elif not isinstance(uncertainty, type(None)) and not isinstance(
            covariance, type(None)
        ):
            msg = (
                f"set method for {label} only allows either uncertainty"
                f"or covariance, not both"
            )
            raise ValueError(msg)

    def set_all(self, data_handler: "DataHandler"):
        """sets all of the vectors and matrices from a DataHandler or child of
        a DataHandler

        Parameters
        ----------
        data_handler : DataHandler
            A data hanlder or child of, from which any settable data
            is transfered from this to the instance of the class
            with settable labels defined in self._settable_attributes
        """
        for label in self._settable_attributes:
            data = getattr(data_handler, f"_{label}")
            if isinstance(data, type(None)):
                continue
            elif f"_{label}_covariance" in self.__dict__:
                covariance = getattr(data_handler, f"_{label}_covariance")
                self.set(label, data, covariance=covariance)
            else:
                self.set(label, data)

    # set, load and save
    @check_vals(
        variable_type={
            "label": is_string,
            "input_path": is_pathlike,
            "uncertainty": is_pathlike,
            "covariance": is_pathlike,
        }
    )
    def load(
        self,
        label: str,
        input_path: Union[str, PathLike],
        uncertainty: Union[str, PathLike] = None,
        covariance: Union[str, PathLike] = None,
    ):
        """loads data from files

        Parameters
        ----------
        label : str
            Label of the vector or matrix for which data is loaded
        input : str or PathLike
            Path to the input file of the data to associate with
            the label
        uncertainty : str or PathLike, optional
            Path to the input file of the uncertainty to associate with
            the label
        covariance : str or PathLike, optional
            Path to the input file of the covariance to asscilate with
            the label
        """
        data = self._serialiser.load(input_path, self._np_dtype)

        if (not isinstance(uncertainty, type(None))) and isinstance(
            covariance, type(None)
        ):
            # case 1: uncertainty and not covariance supplied
            data_uncertainty = self._serialiser.load(uncertainty, self._np_dtype)
            data_covariance = None
        elif (not isinstance(covariance, type(None))) and isinstance(
            uncertainty, type(None)
        ):
            # case 2: covariance and not uncertainty supplied
            data_covariance = self._serialiser.load(covariance, self._np_dtype)
            data_uncertainty = None
        elif not isinstance(covariance, type(None)) and not isinstance(
            uncertainty, type(None)
        ):
            # case 3
            msg = "uncertainty and covariance both specified, only one can " "be loaded"
            raise ValueError(msg)
        else:
            data_uncertainty = None
            data_covariance = None

        self.set(label, data, uncertainty=data_uncertainty, covariance=data_covariance)

    # set, load and save
    @check_vals(
        variable_type={
            "label": is_string,
            "input_path": is_pathlike,
            "uncertainty": is_pathlike,
            "covariance": is_pathlike,
        }
    )
    def save(
        self,
        label: str,
        output: Union[str, PathLike],
        uncertainty: Union[str, PathLike] = None,
        covariance: Union[str, PathLike] = None,
    ):
        """Saves data to files

        Parameters
        ----------
        label : str
            Name of the vector or matrix for which data is saved
        output : str or pathLike
            Path to the output file
        uncertainty : str or pathLike, optional
            Path to the input file
        covariance : str or pathLike, optional
            Path to the input file
        """

        if label in self._savable_attributes:
            data = getattr(self, label)
            data_uncertainty = getattr(self, f"{label}_uncertainty")
        else:
            avail = "\n    ".join(self._settable_attributes)
            msg = (
                f"unable to use general save method for attribute {label} "
                f"available options are {avail}"
            )
            raise KeyError(msg)

        self._serialiser.save(output, data)

        if not isinstance(uncertainty, type(None)):
            self._serialiser.save(uncertainty, data_uncertainty)

    def _set_ndarray(
        self, data: Union[list[float], np.typing.NDArray], dimensions: int
    ):
        """Convert any inputs into required numpy nd array

        Parameters
        ----------
        data : Union[list[float], np.typing.NDArray]
            Iterable containing vector or matrix to be converted to a NDArray
        dimensions : int
            Dimension required for data vector or matrix, used to error check

        Returns
        -------
        array : np.nd_array
            Numpy array containing data
        """

        array = np.array(data, self._np_dtype)
        if dimensions == 1:
            if len(array.shape) == 1:
                return array
            elif len(array.shape) == 2 and array.shape[1] == 1:
                return array.transpose()[0]
            else:
                msg = (
                    f"shape of data not equal to required dimension:"
                    f" shape {array.shape} required dimensions {dimensions}"
                )
                raise ValueError(msg)
        elif len(array.shape) == dimensions:
            return array
        else:
            msg = f"shape of loaded data not a vector: shape {array.shape}"
            raise TypeError(msg)

    # m and n check for congruence. m rows and n cols in response matrix
    @property
    def shape_m(self) -> int:
        """Gets m shape required for congruence, number of rows in response
        matrix, number of measured values etc...

        Returns
        -------
        m : int
            The shape of set vectors or matrices, represents the number of
            rows in the response matrix and the number of measured values
        """
        return self._m

    @shape_m.setter
    @val_passed_is_integer
    def shape_m(self, shape_m: int):
        """Sets the m shape required for congruence, number of rows in
        response matrix, number of measured values etc...

        Parameters
        ----------
        m : int
            The shape of set vectors or matrices, represents the number of
            rows in the response matrix and the number of measured values
        """

        if isinstance(self._m, type(None)):
            self._m = shape_m
        elif self._m != shape_m:
            msg = (
                f"values set are incongruous for m."
                f" {shape_m} passed {self._m} required."
            )
            raise ValueError(msg)
        else:
            pass

    @property
    def shape_n(self):
        """Gets n shape required for congruence, number of columns in response
        matrix, number of bins in spectra etc...

        Returns
        -------
        n : int
            The shape of set vectors or matrices, represents the number of
            columns in the response matrix and the number of bins in spectra
        """
        return self._n

    @shape_n.setter
    @val_passed_is_integer
    def shape_n(self, shape_n):
        """Sets n shape required for congruence, number of columns in response
        matrix, number of bins in spectra etc...

        Parameters
        ----------
        n : int
            The shape of set vectors or matrices, represents the number of
            columns in the response matrix and the number of bins in spectra
        """
        if isinstance(self._n, type(None)):
            self._n = shape_n
        elif self._n != shape_n:
            msg = (
                f"values set are incongruous or n."
                f" {shape_n} passed {self._n} required."
            )
            raise ValueError(msg)
        else:
            pass

    # group structure
    @property
    def group_structure(self):
        """Gets the group_structure attribute

        Returns
        ------
        group_structure : Union[list[float], np.typing.NDArray]
            The group structure of spectra
        """
        return self._group_structure

    @group_structure.setter
    @iterable_passed_is_numeric
    def group_structure(self, group_structure: Union[list[float], np.typing.NDArray]):
        """Sets the group_structure attribute

        Parameter
        ---------
        group_structure : Union[list[float], np.typing.NDArray]
            The group structure of spectra
        """
        self._group_structure = self._set_ndarray(group_structure, 1)
        self.shape_n = self._group_structure.shape[0] - 1

    # a priori attribute getters and setters
    @property
    def a_priori(self):
        """Gets the a_priori attribute

        Returns
        -------
        a_priori : Union[list[float], np.typing.NDArray]
            A priori spectrum
        """
        return self._a_priori

    @a_priori.setter
    @iterable_passed_is_numeric
    def a_priori(self, a_priori: Union[list[float], np.typing.NDArray]):
        """Sets the a_priori attribute

        Parameter
        ---------
        a_priori : Union[list[float], np.typing.NDArray]
            A priori spectrum
        """
        self._a_priori = self._set_ndarray(a_priori, 1)
        self.shape_n = self._a_priori.shape[0]

    @property
    def a_priori_uncertainty(self):
        """Gets the a_priori uncertinaty attribute

        Returns
        -------
        a_priori_uncertainty : Union[list[float], np.typing.NDArray]
            A priori spectrum uncertainty
        """
        return self._a_priori_uncertainty

    @a_priori_uncertainty.setter
    @iterable_passed_is_numeric
    def a_priori_uncertainty(
        self, a_priori_uncertainty: Union[list[float], np.typing.NDArray]
    ):
        """Sets the a_priori uncertainty attribute

        Parameter
        ---------
        a_priori_uncertainty : Union[list[float], np.typing.NDArray]
            A priori spectrum
        """
        self._a_priori_uncertainty = self._set_ndarray(a_priori_uncertainty, 1)

        self.shape_n = self._a_priori.shape[0]
        self._a_priori_covariance = vuc(a_priori_uncertainty)

    @property
    def a_priori_covariance(self):
        """Gets the a_priori covariance attribute

        Returns
        -------
        a_priori_covariance : Union[list[float], np.typing.NDArray]
            A priori spectrum covariance
        """
        return self._a_priori_covariance

    @a_priori_covariance.setter
    @iterable_passed_is_numeric
    def a_priori_covariance(
        self, a_priori_covariance: Union[list[float], np.typing.NDArray]
    ):
        """Sets the a_priori covaraince attribute

        Parameter
        ---------
        a_priori_uncertainty : Union[list[float], np.typing.NDArray]
            A priori spectrum
        """
        self._a_priori_covariance = self._set_ndarray(a_priori_covariance, 2)
        self.shape_n = self._a_priori.shape[0]
        self._a_priori_uncertainty = vcu(a_priori_covariance)

    # response_matrix attribute getters and setters

    @property
    def response_matrix(self):
        """Gets the response_matrix attribute

        Returns
        -------
        response_matrix : Union[list[float], np.typing.NDArray]
            Response matrix
        """
        return self._response_matrix

    @response_matrix.setter
    @iterable_passed_is_numeric
    def response_matrix(self, response_matrix: Union[list[float], np.typing.NDArray]):
        """Sets the response_matrix attribute

        Attribute
        ---------
        response_matrix : Union[list[float], np.typing.NDArray]
            Response matrix
        """
        self._response_matrix = self._set_ndarray(response_matrix, 2)
        self.shape_m = self._response_matrix.shape[0]
        self.shape_n = self._response_matrix.shape[1]

    @property
    def response_matrix_uncertainty(self):
        """Gets the response_matrix_uncertainty attribute

        Returns
        -------
        response_matrix_uncertainty : Union[list[float], np.typing.NDArray]
            Response matrix uncertainty
        """
        return self._response_matrix_uncertainty

    @response_matrix_uncertainty.setter
    @iterable_passed_is_numeric
    def response_matrix_uncertainty(
        self, response_matrix_uncertainty: Union[list[float], np.typing.NDArray]
    ):
        """Sets the response_matrix_uncertainty attribute

        Attribute
        ---------
        response_matrix_uncertainty : Union[list[float], np.typing.NDArray]
            Response matrix uncertainty
        """
        self._response_matrix_uncertainty = self._set_ndarray(
            response_matrix_uncertainty, 2
        )
        self.shape_m = self._response_matrix.shape[0]
        self.shape_n = self._response_matrix.shape[1]
        self._response_matrix_covariance = muc(response_matrix_uncertainty)

    @property
    def response_matrix_covariance(self):
        """Gets the response_matrix_covariance attribute

        Returns
        -------
        response_matrix_covaraince : np.nd_array
            Response matrix covariance
        """
        return self._response_matrix_covariance

    @response_matrix_covariance.setter
    @iterable_passed_is_numeric
    def response_matrix_covariance(
        self, response_matrix_covariance: Union[list[float], np.typing.NDArray]
    ):
        """Sets the response_matrix_covariance attribute

        Parameters
        ----------
        response_matrix_covaraince : Union[list[float], np.typing.NDArray]
            Response matrix covariance
        """
        self._response_matrix_covariance = self._set_ndarray(
            response_matrix_covariance, 4
        )
        self.shape_m = self._response_matrix.shape[0]
        self.shape_n = self._response_matrix.shape[1]
        self.shape_m = self._response_matrix.shape[2]
        self.shape_n = self._response_matrix.shape[3]
        self._response_matrix_uncertainty = mcu(response_matrix_covariance)

    # measured attribute getters and setters
    @property
    def measured(self):
        """Gets the measured values

        Returns
        -------
        measured : Union[list[float], np.typing.NDArray]
            Measured values
        """
        return self._measured

    @measured.setter
    @iterable_passed_is_numeric
    def measured(self, measured: Union[list[float], np.typing.NDArray]):
        """sets the measured values attribute

        Parameters
        ----------
        measured : Union[list[float], np.typing.NDArray]
            Measured values
        """
        self._measured = self._set_ndarray(measured, 1)
        self.shape_m = self._measured.shape[0]

    @property
    def measured_uncertainty(self):
        """Gets the measured values uncertinaty

        Returns
        -------
        measured_uncertainty : Union[list[float], np.typing.NDArray]
            Measured values uncertanties
        """
        return self._measured_uncertainty

    @measured_uncertainty.setter
    @iterable_passed_is_numeric
    def measured_uncertainty(
        self, measured_uncertainty: Union[list[float], np.typing.NDArray]
    ):
        """Sets the measured values uncertinaty

        Parameters
        ----------
        measured_uncertainty : Union[list[float], np.typing.NDArray]
            Measured values uncertanties
        """
        self._measured_uncertainty = self._set_ndarray(measured_uncertainty, 1)
        self.shape_m = self._measured_uncertainty.shape[0]
        self._measured_covariance = vuc(measured_uncertainty)

    @property
    def measured_covariance(self):
        """Gets the measured values covariance

        Returns
        -------
        measured_covariance : Union[list[float], np.typing.NDArray]
            Measured values covariance
        """
        return self._measured_covariance

    @measured_covariance.setter
    @iterable_passed_is_numeric
    def measured_covariance(
        self, measured_covariance: Union[list[float], np.typing.NDArray]
    ):
        """Sets the measured values covariance

        Parameters
        ----------
        measured_covariance : Union[list[float], np.typing.NDArray]
            Measured values covariance
        """
        self._measured_covariance = self._set_ndarray(measured_covariance, 2)
        self.shape_m = self._measured_covariance.shape[0]
        self.shape_m = self._measured_covariance.shape[1]
        self._measured_uncertainty = vcu(measured_covariance)

    # target
    @property
    def target(self):
        """Gets the target spectrum

        Returns
        -------
        target : Union[list[float], np.typing.NDArray]
            Target specutrm
        """
        return self._target

    @target.setter
    @iterable_passed_is_numeric
    def target(self, target: Union[list[float], np.typing.NDArray]):
        """Sets the target spectrum

        Parameters
        ----------
        target : Union[list[float], np.typing.NDArray]
            Target specutrm
        """
        self._target = self._set_ndarray(target, 1)
        self.shape_n = self._target.shape[0]

    @property
    def target_uncertainty(self):
        """Gets the a_priori uncertinaty attribute

        Returns
        -------
        target_uncertainty : Union[list[float], np.typing.NDArray]
            Target spectrum uncertainty
        """
        return self._target_uncertainty

    @target_uncertainty.setter
    @iterable_passed_is_numeric
    def target_uncertainty(
        self, target_uncertainty: Union[list[float], np.typing.NDArray]
    ):
        """Sets the target uncertainty attribute

        Parameter
        ---------
        target_uncertainty : Union[list[float], np.typing.NDArray]
            Target spectrum
        """
        self._target_uncertainty = self._set_ndarray(target_uncertainty, 1)

        self.shape_n = self._a_priori.shape[0]
        self._target_covariance = vuc(target_uncertainty)

    @property
    def target_covariance(self):
        """Gets the target covariance attribute

        Returns
        -------
        target_covariance : Union[list[float], np.typing.NDArray]
            Target spectrum covariance
        """
        return self._a_priori_covariance

    @target_covariance.setter
    @iterable_passed_is_numeric
    def target_covariance(
        self, target_covariance: Union[list[float], np.typing.NDArray]
    ):
        """Sets the target covaraince attribute

        Parameter
        ---------
        target_uncertainty : Union[list[float], np.typing.NDArray]
            Target spectrum
        """
        self._target_covariance = self._set_ndarray(target_covariance, 2)
        self.shape_n = self._a_priori.shape[0]
        self._target_uncertainty = vcu(target_covariance)

    @property
    def method(self):
        """Returns the string associated with the method

        Returns
        -------
        method : str
            A string of the method
        """
        return self._method

    # solution

    @property
    def solution(self):
        """Gets the solution spectrum

        Returns
        -------
        solution : Union[list[float], np.typing.NDArray]
            Solution spectrum
        """
        return self._solution

    @property
    def fitted(self):
        """Gets the fitted measured values associated with the solution

        Returns
        -------
        fitter : Union[list[float], np.typing.NDArray]
            Fitted measured values associated with the solution
        """
        return np.dot(self._response_matrix, self._solution)

    # statistics
    @property
    def structure_matrix(self):
        """Returns the structure matrix of the response matrix"""
        return np.dot(
            np.dot(self.response_matrix.T, self.measured_covariance),
            self.response_matrix,
        )

    @property
    def reduced_chi_squared(self):
        """Returns the reduced chi squared of the solution measured values
        assiciated with the solution

        Returns
        -------
        reduced_chi_squared : float
            Teduced chi squared of the solution measured values
            assiciated with the solution
        """
        # calculate the chi-squared of the measured and fitted values
        chi_sq = ((self._measured - self.fitted) / self._measured_uncertainty) ** 2
        return sum(chi_sq) / self._m

    @property
    def residuals(self):
        """Returns the residuals of the measured and fitted values

        Returns
        -------
        residual : Union[list[float], np.typing.NDArray]
            residuals of the measured and fitted values
        """
        return self.fitted - self._measured

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

    def _run(self, method: str):
        """general run method called by each unfolding methods run method.
        This checks the user is providing a suitable run method string

        Parameters
        ----------
        method : str
            The run methd requred by the unfodling method class.
        """

        # check method is a valid option, else raise exception
        if method not in self._methods:
            avail = "\n    > ".join(self._method)
            msg = (
                f"unable to use run method {method} "
                f"available options are \n\n    > {avail}"
            )
            raise ValueError(msg)

        # run the required method
        self.logger.info(f"running method: {self._method} :: {method}")

        # get the result from the method and run
        return self._methods[method]()
