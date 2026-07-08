import spectrauf as suf
import matplotlib.pyplot as plt
import numpy as np

# load the input data into a generic data handler
data = suf.datahandler.DataHandler()
data.load('measured', 'reaction_rates.csv', uncertainty='reaction_rate_uncertainties.csv')
data.load('response_matrix', 'response_matrix.csv')
data.load('group_structure', 'group_structure.csv')
data.load('a_priori', 'a_priori.csv')
data.load('target', 'target.csv')

# non-linear least squared based on GRAVEL algorithm
PARAMS_NLSQ_B = {'reduced_chi_squared': 1.0, 'max_iterations': 1e3}
nlsq_b_mc = suf.montecarlo.MonteCarlo()
nlsq_b_mc.set_all(data)
nlsq_b_mc.run(suf.leastsquares.NonlinearLeastSquareIterativeB,
              'reduced_chi_squared',
              run_parameters=PARAMS_NLSQ_B,
              batch_size=1,
              min_trials=5,
              confidence=0.10,
              tolerance=0.80,
              parameters_to_randomise=['measured'])

# non-linear least squared based on GRAVEL algorithm
PARAMS_MPE = {'omega': len(data.measured)}
mpe_mc = suf.montecarlo.MonteCarlo()
mpe_mc.set_all(data)
mpe_mc.run(suf.entropy.MaximumPotentialEntropy,
           'basinhopping',
           run_parameters=PARAMS_MPE,
           batch_size=1,
           min_trials=5,
           confidence=0.10,
           tolerance=0.80,
           parameters_to_randomise=['measured'])

# plot results
fig, ax = plt.subplots(1)

# a priori
a_priori_per_lethargy = suf.util.convert_spectrum_to_lethargy(data.a_priori, data.group_structure)
a_priori_extended = np.concatenate((a_priori_per_lethargy, a_priori_per_lethargy[-1:]))
ax.step(data.group_structure, 
        a_priori_extended,
        label='a_priori', 
        linestyle='dashed',
        where='post',
        color='blue')

# target
target_per_lethargy = suf.util.convert_spectrum_to_lethargy(data.target, data.group_structure)
target_extended = np.concatenate((target_per_lethargy, target_per_lethargy[-1:]))
ax.step(data.group_structure, 
        target_extended,
        label='target', 
        where='post',
        color='orange')

# solution with uncertainty (nlsq_b)
solution_nlsq_b_per_lethargy = suf.util.convert_spectrum_to_lethargy(nlsq_b_mc.unfolded_data.solution, nlsq_b_mc.unfolded_data.group_structure)
solution_nlsq_b_extended = np.concatenate((solution_nlsq_b_per_lethargy, solution_nlsq_b_per_lethargy[-1:]))
ax.step(data.group_structure, 
        solution_nlsq_b_extended,
        label='solution (nlsq_b)',
        where='post',
        color='green')

# solution uncertainty (nlsq_b)
sol_nlsq_b_min = nlsq_b_mc.unfolded_data.solution - nlsq_b_mc.monte_carlo_solution_std_dev
sol_nlsq_b_max = nlsq_b_mc.unfolded_data.solution + nlsq_b_mc.monte_carlo_solution_std_dev
sol_nlsq_b_min_per_lethargy = suf.util.convert_spectrum_to_lethargy(sol_nlsq_b_min, nlsq_b_mc.unfolded_data.group_structure)
sol_nlsq_b_max_per_lethargy = suf.util.convert_spectrum_to_lethargy(sol_nlsq_b_max, nlsq_b_mc.unfolded_data.group_structure)
sol_nlsq_b_min_per_lethargy_extended = np.concatenate((sol_nlsq_b_min_per_lethargy, sol_nlsq_b_min_per_lethargy[-1:]))
sol_nlsq_b_max_per_lethargy_extended = np.concatenate((sol_nlsq_b_max_per_lethargy, sol_nlsq_b_max_per_lethargy[-1:]))
ax.fill_between(data.group_structure, 
                sol_nlsq_b_min_per_lethargy_extended,
                sol_nlsq_b_max_per_lethargy_extended,
                step='post', 
                alpha=0.25, 
                label=r'solution uncertainty 1 $\sigma$ (nlsq_b)',
                color='green')

# solution with uncertainty (nlsq_b)
solution_mpe_per_lethargy = suf.util.convert_spectrum_to_lethargy(mpe_mc.unfolded_data.solution, mpe_mc.unfolded_data.group_structure)
solution_mpe_extended = np.concatenate((solution_mpe_per_lethargy, solution_mpe_per_lethargy[-1:]))
ax.step(data.group_structure,
        solution_mpe_extended,
        label='solution (mpe_mc)',
        where='post',
        color='red')

# solution uncertainty
sol_mpe_mc_min = mpe_mc.unfolded_data.solution - mpe_mc.monte_carlo_solution_std_dev
sol_mpe_mc_max = mpe_mc.unfolded_data.solution + mpe_mc.monte_carlo_solution_std_dev
sol_mpe_min_per_lethargy = suf.util.convert_spectrum_to_lethargy(sol_mpe_mc_min, mpe_mc.unfolded_data.group_structure)
sol_mpe_max_per_lethargy = suf.util.convert_spectrum_to_lethargy(sol_mpe_mc_max, mpe_mc.unfolded_data.group_structure)
sol_mpe_min_per_lethargy_extended = np.concatenate((sol_mpe_min_per_lethargy, sol_mpe_min_per_lethargy[-1:]))
sol_mpe_max_per_lethargy_extended = np.concatenate((sol_mpe_max_per_lethargy, sol_mpe_max_per_lethargy[-1:]))
ax.fill_between(data.group_structure, 
                sol_mpe_min_per_lethargy_extended,
                sol_mpe_max_per_lethargy_extended,
                step='post', 
                alpha=0.25, 
                label=r'solution uncertainty 1 $\sigma$ (mpe_mc)',
                color='red')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('energy (MeV)')
ax.set_ylabel('flux ($cm^{-2}$)')
ax.grid()
ax.legend()
plt.show()
