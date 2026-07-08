import spectrauf as suf
import matplotlib.pyplot as plt
import numpy as np

# load the data into a general data handler
data = suf.datahandler.DataHandler()
data.load('measured', 'reaction_rates.csv',
                uncertainty='reaction_rate_uncertainties.csv')
data.load('response_matrix', 'response_matrix.csv')
data.load('group_structure', 'group_structure.csv')
data.load('a_priori', 'a_priori.csv')
data.load('target', 'target.csv')

# non-linear least squared based on GRAVEL algorithm
nlsq_b = suf.leastsquares.NonlinearLeastSquareIterativeB()
nlsq_b.set_all(data)
nlsq_b.run(method='reduced_chi_squared', reduced_chi_squared=1.0)

# entropy method based on MAXED
mpe = suf.entropy.MaximumPotentialEntropy()
mpe.set_all(data)
mpe.run(method='basinhopping', omega=len(data.measured))
print('lambda values')
print(mpe.lambdas)

# plot results
fig, ax = plt.subplots(1)

# plot a priori 
a_priori_per_lethargy = suf.util.convert_spectrum_to_lethargy(data.a_priori, data.group_structure)
a_priori_extended = np.concatenate((a_priori_per_lethargy, a_priori_per_lethargy[-1:]))
ax.step(data.group_structure, 
        a_priori_extended, 
        label='a_priori',)

# plot a priori (mpe scaled)
a_priori_per_lethargy = suf.util.convert_spectrum_to_lethargy(mpe.a_priori, mpe.group_structure)
a_priori_mpe_extended = np.concatenate((a_priori_per_lethargy, a_priori_per_lethargy[-1:]))
ax.step(data.group_structure, 
        a_priori_mpe_extended, 
        label='a priori (mpe scaled)')

# plot target
target_per_lethargy = suf.util.convert_spectrum_to_lethargy(data.target, data.group_structure)
target_extended = np.concatenate((target_per_lethargy, target_per_lethargy[-1:]))
ax.step(data.group_structure, 
        target_extended, 
        label='target',
        ls='--')

# plot solution (nlsq_b)
solution_per_lethargy = suf.util.convert_spectrum_to_lethargy(nlsq_b.solution, nlsq_b.group_structure)
solution_nlsq_b_extended = np.concatenate((solution_per_lethargy, solution_per_lethargy[-1:]))
ax.step(data.group_structure, 
        solution_nlsq_b_extended, 
        label='solution (nlsq_b)')

# plot solution (mpe)
solution_per_lethargy = suf.util.convert_spectrum_to_lethargy(mpe.solution, mpe.group_structure)
solution_mpe_extended = np.concatenate((solution_per_lethargy, solution_per_lethargy[-1:]))
ax.step(data.group_structure, 
        solution_mpe_extended, 
        label='solution (mpe)')

# plot solution (mpe iterative)
solution_per_lethargy = suf.util.convert_spectrum_to_lethargy(mpe.solution, mpe.group_structure)
solution_mpe_extended = np.concatenate((solution_per_lethargy, solution_per_lethargy[-1:]))
ax.step(data.group_structure, 
        solution_mpe_extended, 
        label='solution (mpe)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('energy (MeV)')
ax.set_ylabel('flux ($cm^{-2}$)')
ax.grid()
ax.legend()
plt.show()
