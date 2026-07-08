==========
User guide
==========

Introduction
------------

The SPECTRA-UF code provides a suite of unfolding algorithms as part of a Python API which includes pre and post processing to allow the automation and data visualisation of data sets. 

The unfolding methods available are listed below in :numref:`table_unfolding_methods`.

.. _table_unfolding_methods:
.. list-table:: Unfolding Methods Summary
   :widths: 20 50 30
   :header-rows: 1

   * - Method
     - Description
     - References
   * - **NonlinearLeastSquareIterativeA**
     - An non-linear least square method which iteratively adjusts a prior distribution in order to minimize the difference in squares between the measured and the calculated values. This is based on the underlying mathematics of the United States Air Force (USAF) code SAND-II, released in 1967.
     - A computer-automated iterative method for neutron flux spectra determination by foil activatoin. W. N. McElroy, et al. USAF report. 1967.
   * - **NonlinearLeastSquareIterativeB**
     - An non-linear least square method which iteratively adjusts a prior distribution in order to minimize the difference in squares between the measured and the calculated values. This is based on the underlying mathematics of the HEPRO code GRAVEL, written my M. Matzke.
     - Propagation of uncertainties in unfolding procedures. M. Matzke. Nuclear Instruments and Methods in Physical Research A. 2002. Vol 476. P230-243.
   * - **NonlinearLeastSquareIterativeC**
     - An non-linear least square method which iteratively adjusts a prior distribution in order to minimize the difference in squares between the measured and the calculated values. This is based on the underlying mathematics of the SPUNIT code, written my L. W. Brackenbush and R. I. Scherpelz in 1984.
     - A computer code for multisphere unfolding. L. W. Brackenbush, R. I. Scherpelz. Pacific Northwest Laboratories (PNL) report. 1984.
   * - **MaximumPotentialEntropy**
     - An algorithm which aims to find a solution distribution by minimizing both the squares of the measured and calculated values, along with the minimization of the Kullback-Leibler divergence of the a priori and the solution distributions. This is based on the underlying mathematics of the MAXED code written by M. Reginatto and P. Goldhagen in 1998.
     - MAXED, a computer code for the decomvoution of multisphere neutron spectrometer data using the maximum entropy method. M. Reginatto and P. Goldhagen. Environmental Measurements Laboratory (EML) report, 1998. 
   * - **Parametric**
     - A parametric method which sums together known physical distributions found within distributions, such as Watt spectra, Maxwellian and Gaussian distributions, and optimises the parameters of each distributoin to minimize the square of the calculated and measured values.
     - Custom algorithm, written by R. Worrall.

Data handlers
-------------

The data provided to each unfolding method can be provided via text files (comma or space 
seperated value, csv) or set via a .set method. There is a generic DataHandler class which 
has a common interface with all unfolding methods, and can be used to load data once and 
then pass the data to a range of unfolding methods. Where sets of data have uncertainties 
or covariances attached, these can optionally be loaded at the same time. An example can 
be found below in :numref:`code_data_handler`.

.. _code_data_handler:
.. code-block:: python
   :linenos:
   :caption: Setting up a generic data handler

   import spectrauf as suf
   import matplotlib.pyplot as plt
   
   # intialise the data handler and load data
   data = suf.datahandler.DataHandler()
   data.load('a_priori', 'path/to/a_priori.csv')
   data.load('group_structure', 'path/to/group_structure.csv')
   data.load('response_matrix', 'path/to/response_matrix.csv')
   data.load('measured', 'path/to/measured.csv', uncertainties='path/to/measured_uncertainties.csv')

   # we can access the data and plot the spectra privded.
   fig, ax = plt.subplots(1)
   ax.step(data.group_structure[0], data.a_priori)
   ax.set_xlabel('Energy (MeV)')
   ax.set_ylabel('Intensity (arb. units)')
   plt.show()

Utilities
---------

SPECTRA-UF contains a set of utilities which can be used to pre and post process data. 

See API docs: :mod:`spectrauf.util`

* re-bining of a histogram, either using a linear or log space division method.
* conversion of histrograms to and from per unit methargy

Example usage of these are shown in the code below :numref:`code_rebinning_and_lethargy`.

Re-bin
``````

See API docs: rebin :func:`spectrauf.util.rebin`.

The re-binning method can be used to change the bin boundaries of a histogram, but care should be
taken not to try and re-bin to a finer set of bins as information can be lost.

The re-binning method also allows the division of bins via either linear space (deafult) or via 
logarithmic space for non-linearly spaced hisogram boundaries.

Per unit lethargy conversion
````````````````````````````

When plotting and comparing histograms, such as flux spectra, it is often useful to convert 
histogram distributions into units of per unit lethargy. This allows a visualisation of the distribution taking
into account the widths of unequal hisogram bounds. The definiton of lethargy is shown in equation 
:eq:`eq_lethargy`, where U is the lethargy of a given hisogram bin, and :math:`B_{lower}` and :math:`B_{upper}` 
are the lower and upper bounds of the bin.

.. math:: U = ln(\frac{B_{upper}}{B_{lower}})
   :label: eq_lethargy

Example code
````````````

The below code shows how to apply the re-binning method and then to plot the two distributions in units of per unit lethargy.

.. _code_rebinning_and_lethargy:
.. literalinclude:: /../../examples/example_data_set_3/example_3a_rebinning.py
   :language: python
   :caption: examples/example_data_set_3/example_3a_rebinning.py
   :name: example_set_3_example_3a

Unfolding methods
-----------------

Non-linear least squares
````````````````````````

The non-linear least square method **NonlinearLeastSquareIterativeB** requires an 
initial *a priori* distribution along with a set of *measured* values and their 
*uncertainties*, and a *reponse matrix*.

This is an iterative method and is purely deterministic, and will run until a user defined criteria
is met. By default this is a reduced chi-squeared per degree of freedeom on unity between the measured
and the calculated values, but can be set to a fixed number of iterations.

An example of the NonlinearLeastSquareIterativeB methodology can be found below. 

.. literalinclude:: /../../examples/example_data_set_1/example_1a.py
   :language: python
   :caption: examples/example_data_set_1/example_1a.py
   :name: example_set_1_example_1a

Maximum entropy
```````````````

The maximum entropy method **MaximumPotentialEntropy** requires an initial 
*a priori* distribution along with a set of *measured* values and their *uncertainties*, and a *response matrix*. 

.. literalinclude:: /../../examples/example_data_set_1/example_1b.py
   :language: python
   :caption: examples/example_data_set_1/example_1b.py
   :name: example_set_1_example_1b
