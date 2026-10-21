Installation
============

SPECTRA-UF requires Python 3.12 or newer. The current release is a **beta**
(|release|), so the API may change between minor versions.

Install a release
-----------------

Releases are tagged on GitHub. To install a specific release:

.. _bash_install_end_user:
.. code-block:: bash
   :linenos:
   :caption: installing a tagged release of SPECTRA-UF

   python -m pip install git+https://github.com/ukaea/spectra-uf.git@v0.1.0b1

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