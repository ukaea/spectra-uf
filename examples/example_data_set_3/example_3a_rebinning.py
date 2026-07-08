import spectrauf as suf
import matplotlib.pyplot as plt
import numpy as np

# load data using a generic data handler
data = suf.datahandler.DataHandler()
data.load('a_priori', 'a_priori.csv')
data.load('group_structure', 'group_structure.csv')

# set up new group structure
group_structure_new = np.logspace(-2, 2, 50)

# re-bin the histogram
a_priori_rebinned = suf.util.rebin(
    data.a_priori,           # input hisogram values
    data.group_structure,    # boundaries of histogram
    group_structure_new      # new bounds required
)

a_priori_per_unit_lethargy = suf.util.convert_spectrum_to_lethargy(
    data.a_priori, 
    data.group_structure
)

a_priori_rebinned_per_unit_lethargy = suf.util.convert_spectrum_to_lethargy(
    a_priori_rebinned, 
    group_structure_new
)

# set up a figure and axis
fig, ax = plt.subplots(1)

# add a spectrum as a step plot for the original spectrum
ax.step(
    data.group_structure[:-1],
    a_priori_per_unit_lethargy, 
    label=f'flux in {len(data.group_structure)-1} bins', 
    where='post')

# add a spectrum as a step plot for the re-binned spectrum
ax.step(
    group_structure_new[:-1], 
    a_priori_rebinned_per_unit_lethargy, 
    label=f'flux re-binned in {len(group_structure_new)-1} bins',
    where='post')

# set the scale from the default linear to logarithmic ('lin' or 'log' string passed as argument)
ax.set_xscale('log')
ax.set_yscale('log')

# set some axis labels
ax.set_xlabel('Energy (MeV)')
ax.set_ylabel('Flux per unit lethargy ($cm^{-2}$)')

# add grid lines
ax.grid()
ax.legend()

# set a title
ax.set_title('FISPACT-II Reference spectra: NIF Ignition')

# show the figure
plt.show()