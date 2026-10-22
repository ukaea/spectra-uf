Installation
============

SPECTRA-UF requires Python 3.12 or newer. The current version is a **beta**
(|release|), so the API may change between minor versions.

Install
-------

To install directly from GitHub:

.. _bash_install_end_user:
.. code-block:: bash
   :linenos:
   :caption: installing SPECTRA-UF from GitHub

   python -m pip install git+https://github.com/ukaea/spectra-uf.git

Or to install from a local clone of the default branch:

.. code-block:: bash
   :linenos:
   :caption: installing SPECTRA-UF from a clone

   git clone https://github.com/ukaea/spectra-uf.git
   cd spectra-uf
   python -m pip install .

Development install
-------------------

For those wishing to use the latest development version or contribute to the
code or docs, clone the ``dev`` branch and install in editable mode with the
development and documentation extras:

.. _bash_install_dev_and_docs:
.. code-block:: bash
   :linenos:
   :caption: installing SPECTRA-UF for development

   git clone -b dev https://github.com/ukaea/spectra-uf.git
   cd spectra-uf
   python -m pip install -e ".[dev,docs]"

The quotes around ``".[dev,docs]"`` are needed in zsh (the macOS default shell).