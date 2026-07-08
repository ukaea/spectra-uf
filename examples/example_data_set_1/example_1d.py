import spectrauf as suf
import matplotlib.pyplot as plt
import numpy as np


# create instance of unfolding method and load data
data = suf.datahandler.DataHandler()
data.load('response_matrix', 'response_matrix.csv')
data.load('measured', 'reaction_rates.csv',
            uncertainty='reaction_rate_uncertanties.csv')
data.load('a_priori', 'a_priori.csv')
data.load('group_structure', 'group_structure.csv')
data.load('target', 'target.csv')

RUN_METHOD = 'reduced_chi_squared'
RUN_PARAMS = {'reduced_chi_squared': 1.0,
              'max_iterations': 1e4}

# run single case
nlsq_b_mc = suf.montecarlo.MonteCarlo()
nlsq_b_mc.set_all(data)
nlsq_b_mc.run(suf.leastsquares.NonlinearLeastSquareIterativeB, 
              RUN_METHOD,
              RUN_PARAMS,
              batch_size=25,
              min_trials=250,
              confidence=0.95,
              tolerance=0.10,
              parameters_to_randomise=['measured'])

print(nlsq_b_mc.unfolded_data.solution)
print(nlsq_b_mc.monte_carlo_solution)
print(nlsq_b_mc.monte_carlo_solution_std_dev)

# plot results
fig, ax = plt.subplots(1)

# plot a priori
a_priori_extended = np.concatenate((nlsq_b_mc.unfolded_data.a_priori, 
                                    nlsq_b_mc.unfolded_data.a_priori[-1:]))
ax.step(nlsq_b_mc.unfolded_data.group_structure, 
        a_priori_extended, 
        label='a_priori', 
        where='post')

# plot target
target_extended = np.concatenate((nlsq_b_mc.unfolded_data.target, 
                                  nlsq_b_mc.unfolded_data.target[-1:]))
ax.step(nlsq_b_mc.unfolded_data.group_structure, 
        target_extended, 
        label='target', 
        where='post', 
        linestyle='dashed')

# plot solution
ax.step(nlsq_b_mc.unfolded_data.group_structure, 
        np.concatenate((nlsq_b_mc.unfolded_data.solution, 
                        nlsq_b_mc.unfolded_data.solution[-1:])), 
        label='solution', 
        where='post')

# plot solution uncertainty
solution_min = nlsq_b_mc.unfolded_data.solution - nlsq_b_mc.monte_carlo_solution_std_dev
solution_max = nlsq_b_mc.unfolded_data.solution + nlsq_b_mc.monte_carlo_solution_std_dev
ax.fill_between(nlsq_b_mc.unfolded_data.group_structure, 
                np.concatenate((solution_min, solution_min[-1:])), 
                np.concatenate((solution_max, solution_max[-1:])), 
                step='post', 
                alpha=0.25)

ax.set_xlabel('Energy (arb. units)')
ax.set_ylabel('flux (arb. units)')
ax.grid()
ax.legend()
plt.show()
