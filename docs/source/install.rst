Installation
============

* Install guide

SPECTRA-UF can be installed from the Git repository


.. _bash_install_end_user:
.. code-block:: bash
   :linenos:
   :caption: installing SPECTRA-UF

    git pull git@git.ccfe.ac.uk:rworrall/spectra-uf.git
    cd spectra-uf
    python -m pip install .


For those wishing to use the latest version or contribute to dev / docs

.. _bash_install_dev_and_docs:
.. code-block:: bash
   :linenos:
   :caption: installing SPECTRA-UF

    git pull -b dev git@git.ccfe.ac.uk:rworrall/spectra-uf.git
    cd spectra-uf
    python -m pip install .[dev,docs]