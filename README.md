# SPECTRA-UF

[![Publish docs and badges](https://github.com/ukaea/spectra-uf/actions/workflows/publish-docs.yml/badge.svg?branch=main)](https://github.com/ukaea/spectra-uf/actions/workflows/publish-docs.yml)
[![Coverage](https://ukaea.github.io/spectra-uf/coverage.svg)](https://github.com/ukaea/spectra-uf/actions/workflows/publish-docs.yml)
[![Pylint score](https://ukaea.github.io/spectra-uf/pylint.svg)](https://github.com/ukaea/spectra-uf/actions/workflows/publish-docs.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://ukaea.github.io/spectra-uf/)
[![Release](https://img.shields.io/github/v/release/ukaea/spectra-uf?include_prereleases)](https://github.com/ukaea/spectra-uf/releases)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://github.com/ukaea/spectra-uf/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

SPECTRA-UF provides a suite of spectrum unfolding algorithms as a Python API, including pre- and
post-processing utilities to automate the analysis and visualisation of data sets.

> **Note:** SPECTRA-UF is currently in beta. The API may change between releases; see the
> [changelog](CHANGELOG.md) for details.

Benchmarking of the methods was undertaken using synthetic data sets based on activation foil
detectors and spectra found in a variety of fusion environments.

## Documentation

The documentation is available at <https://ukaea.github.io/spectra-uf/>.

## Installation

SPECTRA-UF requires Python 3.12 or newer. To install directly from GitHub:

```bash
python -m pip install git+https://github.com/ukaea/spectra-uf.git
```

Or to install from a local clone:

```bash
git clone https://github.com/ukaea/spectra-uf.git
cd spectra-uf
python -m pip install .
```

For a development install, see the
[installation guide](https://ukaea.github.io/spectra-uf/install.html).

## Citation

Any use of SPECTRA-UF should reference the paper
*The development, testing and comparison of unfolding methods in SPECTRA-UF for neutron spectrometry*,
<https://doi.org/10.1016/j.fusengdes.2020.112038>.

## Disclaimer

Neither the UK Government, UKAEA or CCFE make any warranty, express or implied, or assume any legal
liability or responsibility for the accuracy, completeness or usefulness of any information,
apparatus, product, or process disclosed, or represents that its use would not infringe privately
owned rights.