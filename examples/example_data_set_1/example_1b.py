import spectrauf as suf
import matplotlib.pyplot as plt
import numpy as np


# create instance of unfolding method and load data
entropy = suf.entropy.MaximumPotentialEntropy()
entropy.load('response_matrix', 'response_matrix.csv')
entropy.load('measured', 'reaction_rates.csv', uncertainty='reaction_rate_uncertanties.csv')
entropy.load('a_priori', 'a_priori.csv')
entropy.load('group_structure', 'group_structure.csv')
entropy.load('target', 'target.csv')

# run
entropy.run()

# plot results
fig, ax = plt.subplots(1)
ax.step(entropy.group_structure,
        np.concatenate((entropy.solution, entropy.solution[-1:])), 
        label='solution', 
        where='post')
ax.step(entropy.group_structure, 
        np.concatenate((entropy.a_priori, entropy.a_priori[-1:])), 
        label='a_priori', 
        where='post')
ax.step(entropy.group_structure, 
        np.concatenate((entropy.target, entropy.target[-1:])), 
        label='target', 
        where='post', 
        ls='dashed')
ax.set_xlabel('Energy (arb. units)')
ax.set_ylabel('flux (arb. units)')
ax.grid()
ax.legend()
plt.show()
