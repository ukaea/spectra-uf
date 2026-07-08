"""Uncertainties contains functions for processing uncertainty quantities"""

import numpy as np
from numpy.linalg import inv as np_inv


def vector_uncertainty_to_cov(uncertainties):
    """Converts the uncertanties of a vector into the covariance matrix
    of a vector
    """
    return np.diag(uncertainties) ** 2


def vector_cov_to_uncertainty(covariance):
    """Converts the covariance matrix of a vector into the uncertainty
    of a vector
    """
    return np.diag(covariance) ** 0.5


def vector_cov_to_rel_cov(vector, covariance):
    """Converts the covariance matrix of a vector into the relative
    covariance of a vector
    """
    vec_diag_inv = np_inv(np.diag(vector))
    return np.dot(np.dot(vec_diag_inv, covariance), vec_diag_inv)


def vector_rel_cov_to_cov(vector, rel_covariance):
    """Converts the relative covariance of a vector into the covariance
    of a vector
    """
    vec_diag = np.diag(vector)
    return np.dot(np.dot(vec_diag, rel_covariance), vec_diag)


def matrix_uncertainty_to_cov(uncertinaty):
    """Converts the uncertainty on a matrix to the covariance matrix of a
    matrix
    """
    cov = np.zeros(np.concatenate((uncertinaty.shape, uncertinaty.shape)))
    for i, row in enumerate(uncertinaty):
        for j, val in enumerate(row):
            cov[i][j][i][j] = val**2
    return cov


def matrix_cov_to_uncertainty(covariance):
    """Converts the covariance matrix of a matrix to the uncertainty of a
    matrix
    """
    unc = np.zeros(covariance.shape[:2])
    for i, row in enumerate(covariance):
        for j, m in enumerate(row):
            unc[i][j] = m[i][j] ** 0.5
    return unc


def matrix_diagonal(matrix):
    """Converts a matrix into the diagonal matrix fulled with zeros"""
    diag_matrix = np.zeros(np.concatenate((matrix.shape, matrix.shape)))
    for i, row in enumerate(matrix):
        for j, _ in enumerate(row):
            diag_matrix[i][j][i][j] = matrix[i][j]
    return diag_matrix
