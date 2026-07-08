import spectrauf as suf
import matplotlib.pyplot as plt
import numpy as np


# constants
UNFOLDING_METHODS = [suf.leastsquares.NonlinearLeastSquareIterativeA,
                     suf.leastsquares.NonlinearLeastSquareIterativeB,
                     suf.leastsquares.NonlinearLeastSquareIterativeC,
                     suf.entropy.MaximumPotentialEntropy]

RUN_METHODS = ['reduced_chi_squared',
               'reduced_chi_squared',
               'reduced_chi_squared',
               'minimize']

RUN_PARAMETERS = [{'reduced_chi_squared': 1.0},
                  {'reduced_chi_squared': 1.0},
                  {'reduced_chi_squared': 1.0},
                  {'omega': 0.25}]

# create generic data instance and load data
data = suf.datahandler.DataHandler()
data.load('response_matrix', 'response_matrix.csv')
data.load('measured', 'reaction_rates.csv',
            uncertainty='reaction_rate_uncertanties.csv')
data.load('a_priori', 'a_priori.csv')
data.load('group_structure', 'group_structure.csv')
data.load('target', 'target.csv')

# plot results
fig, ax = plt.subplots(1)
ax.step(data.group_structure, np.concatenate((data.a_priori, data.a_priori[-1:])), label='a_priori', where='post')
ax.step(data.group_structure, np.concatenate((data.target, data.target[-1:])), label='target', where='post', ls='dashed')

# run each unfolding method
for unfolding_method, run_method, run_param  in zip(UNFOLDING_METHODS, RUN_METHODS, RUN_PARAMETERS):
    # load data into unfolding method
    unfolding_data = unfolding_method()
    unfolding_data.set_all(data)
    unfolding_data.run(method=run_method, **run_param)
    ax.step(unfolding_data.group_structure, np.concatenate((unfolding_data.solution, unfolding_data.solution[-1:])), label=f'{unfolding_data._method}', where='post')

ax.grid()
ax.set_xlabel('Energy (arb. units)')
ax.set_ylabel('flux (arb. units)')

ax.legend()
plt.show()
