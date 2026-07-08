"""
Utilities for validating parameters passed to methods.
"""

import typing
from typing import Union, Any, Callable, Dict, List
import functools
from functools import wraps
import numbers
import os
import inspect

import numpy as np
from numpy.typing import NDArray


def is_numeric(value: Any) -> bool:
    """Check if a value is a numeric type.

    Parameters
    ----------
    value : Any
        The value to validate.

    Returns
    -------
    bool
        True if value is a number (int, float, np.number), False otherwise.
    """
    return isinstance(value, numbers.Number)


def is_integer(value: Any) -> bool:
    """Check if a value is an integer type.

    Parameters
    ----------
    value : Any
        The value to validate.

    Returns
    -------
    bool
        True if value is an integral type, False otherwise.
    """
    return isinstance(value, numbers.Integral)


def is_float(value: Any) -> bool:
    """Check if a value is a real number (float-like).

    Parameters
    ----------
    value : Any
        The value to validate.

    Returns
    -------
    bool
        True if value is a real number, False otherwise.
    """
    return isinstance(value, numbers.Real)

def is_string(value: Any) -> bool:
    """Check if a value is a string.

    Parameters
    ----------
    value : Any
        The value to validate.

    Returns
    -------
    bool
        True if value is a string, False otherwise.
    """
    return isinstance(value, str)

def is_pathlike(value: Any) -> bool:
    """Check if a value is a path-like object.

    Parameters
    ----------
    value : Any
        The value to validate.

    Returns
    -------
    bool
        True if value is str, bytes, or os.PathLike, False otherwise.
    """
    return isinstance(value, (str, bytes, os.PathLike))

def _create_iterable_checker(single_checker: Callable) -> Callable:
    """Factory to create a checker for collections of a specific type (recursive)."""

    def checker(value: Any) -> bool:
        if not isinstance(value, (list, tuple, set, np.ndarray)):
            return False

        if isinstance(value, np.ndarray):
            if single_checker in (is_numeric, is_float, is_integer):
                return np.issubdtype(value.dtype, np.number)
            return all(single_checker(item) for item in value.flat)

        for item in value:
            if isinstance(item, (list, tuple)):
                if not checker(item):
                    return False
            else:
                if not single_checker(item):
                    return False
        return True

    name = single_checker.__name__.replace("is_", "")
    checker.__name__ = f"is_iterable_{name}"
    return checker

# iterable Checkers (boolean functions)
is_iterable_numeric = _create_iterable_checker(is_numeric)
is_iterable_integer = _create_iterable_checker(is_integer)
is_iterable_float = _create_iterable_checker(is_float)
is_iterable_string = _create_iterable_checker(is_string)
is_iterable_pathlike = _create_iterable_checker(is_pathlike)


def _get_type_name(value: Any) -> str:
    """Get a descriptive string of a value's type, including NumPy dtypes."""
    base_type = type(value).__name__
    if hasattr(value, "dtype"):
        return f"{base_type} (dtype: {value.dtype})"
    return base_type


def _create_decorator(checker_func: Callable) -> Callable:
    """Factory to create a decorator that works for both functions and methods."""

    @wraps(checker_func)
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            params = list(inspect.signature(func).parameters.values())
            is_method = params and params[0].name in ("self", "cls")

            arg_to_check = args[1] if is_method else args[0]

            if not checker_func(arg_to_check):
                expected = checker_func.__name__.replace("is_", "").replace("_", " ")
                actual = _get_type_name(arg_to_check)
                raise TypeError(
                    f"Function '{func.__name__}' expected {expected}, "
                    f"but received {actual}."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


# single value decorators
val_passed_is_numeric = _create_decorator(is_numeric)
val_passed_is_integer = _create_decorator(is_integer)
val_passed_is_float = _create_decorator(is_float)
val_passed_is_string = _create_decorator(is_string)
val_passed_is_pathlike = _create_decorator(is_pathlike)

# iterable decorators
iterable_passed_is_numeric = _create_decorator(is_iterable_numeric)
iterable_passed_is_integer = _create_decorator(is_iterable_integer)
iterable_passed_is_float = _create_decorator(is_iterable_float)
iterable_passed_is_string = _create_decorator(is_iterable_string)
iterable_passed_is_pathlike = _create_decorator(is_iterable_pathlike)


def check_vals(
    variable_type: Dict[str, Callable] = None, variable_options: Dict[str, List] = None
) -> Callable:
    """Decorator to validate types and allowed options for multiple arguments.

    Parameters
    ----------
    variable_type : Dict[str, Callable], optional
        Dictionary mapping argument names to checker functions.
    variable_options : Dict[str, List], optional
        Dictionary mapping argument names to lists of allowed values.

    Returns
    -------
    Callable
        The decorated function.

    Raises
    ------
    TypeError
        If an argument does not match the required type.
    ValueError
        If an argument is not among the allowed options or if a decorator
        was passed instead of a checker.
    """
    variable_type = variable_type or {}
    variable_options = variable_options or {}

    for var, checker in variable_type.items():
        if hasattr(checker, "__name__") and "passed_is" in checker.__name__:
            raise ValueError(
                f"Invalid checker for '{var}'. You passed a decorator "
                f"({checker.__name__}) instead of a boolean checker function. "
                f"Use 'is_...' instead of 'val_passed_is_...'."
            )

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for var_name, value in bound_args.arguments.items():
                if var_name in variable_type and value is not None:
                    checker = variable_type[var_name]
                    result = checker(value)

                    if callable(result):
                        raise TypeError(
                            f"Checker for '{var_name}' returned a function instead of a boolean. "
                            "Ensure you are passing a checker like 'is_integer' and not a decorator."
                        )

                    if not result:
                        expected = checker.__name__.replace("is_", "").replace("_", " ")
                        actual = _get_type_name(value)
                        raise TypeError(
                            f"Function '{func.__name__}' argument '{var_name}' "
                            f"expected {expected}, but got {actual}."
                        )

                if var_name in variable_options and value is not None:
                    options = variable_options[var_name]
                    if value not in options:
                        raise ValueError(
                            f"Function '{func.__name__}' argument '{var_name}' "
                            f"received invalid option '{value}'. "
                            f"Must be one of: {options}"
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def is_monotonically_ascending(array: Union[list, NDArray]) -> bool:
    """Check if an array is monotonically ascending.

    Parameters
    ----------
    array : Union[list, NDArray]
        The sequence to check.

    Returns
    -------
    bool
        True if every element is strictly greater than the previous one.
    """
    if len(array) < 2:
        return True

    # efficient check using numpy if available
    if isinstance(array, np.ndarray):
        return np.all(np.diff(array) > 0)

    for lower, upper in zip(array[:-1], array[1:]):
        if lower >= upper:
            return False
    return True
