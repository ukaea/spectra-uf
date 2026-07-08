import spectrauf as suf
import matplotlib.pyplot as plt
import numpy as np


# create instance of unfolding method and load data
nlsq_b = suf.leastsquares.NonlinearLeastSquareIterativeB()
nlsq_b.load('response_matrix', 'response_matrix.csv')
nlsq_b.load('measured', 'reaction_rates.csv',
            uncertainty='reaction_rate_uncertanties.csv')
nlsq_b.load('a_priori', 'a_priori.csv')
nlsq_b.load('group_structure', 'group_structure.csv')
nlsq_b.load('target', 'target.csv')

# run
nlsq_b.run('reduced_chi_squared', reduced_chi_squared=1.0)
# nlsq_b.run(method='iterations', max_iterations=1000)

# plot results
fig, ax = plt.subplots(1)
ax.step(nlsq_b.group_structure,
        np.concatenate((nlsq_b.solution, nlsq_b.solution[-1:])), 
        label='solution', 
        where='post')
ax.step(nlsq_b.group_structure, 
        np.concatenate((nlsq_b.a_priori, nlsq_b.a_priori[-1:])), 
        label='a_priori', 
        where='post')
ax.step(nlsq_b.group_structure, 
        np.concatenate((nlsq_b.target, nlsq_b.target[-1:])), 
        label='target', 
        where='post', 
        ls='dashed')
ax.set_xlabel('Energy (arb. units)')
ax.set_ylabel('flux (arb. units)')
ax.grid()
ax.legend()
plt.show()
