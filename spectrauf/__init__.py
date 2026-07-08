"""
import modules and define data constants at the top level.
"""

from os.path import split, abspath, join
import json
from dataclasses import dataclass
import numpy as np

from . import datahandler
from . import leastsquares
from . import entropy
from . import serialise
from . import stats
from . import logger
from . import montecarlo
from . import parametric
from . import util
from . import regularization
from . import data

# set up logging
logger.setup_logging()

# set constants for file paths to be loaded
_MODULE_ROOT = abspath(split(__file__)[0])
_FIPASCT_REF_SPEC_FILEPATH = join(
    _MODULE_ROOT, "data", "fispact_reference_spectra", "fispact_reference_spectra.json"
)
_FIPASCT_GROUPS_FILEPATH = join(_MODULE_ROOT, "data", "group_structures.json")
_MT_REACTIONS_FILEPATH = join(_MODULE_ROOT, "data", "mt_reactions.json")

# load and store FISPACT-II reference spectra
FISPACT_REFRENCE_SPECTRA = {}
with open(_FIPASCT_REF_SPEC_FILEPATH, "r", encoding="utf8") as input_file:
    spectrum_data_raw = json.load(input_file)

for spectrum_label, spectrum_datum in spectrum_data_raw.items():
    spctrum_datum_stored = util.FispactReferenceSpectrum(
        np.array(spectrum_datum["group_structure"]),
        np.array(spectrum_datum["flux"]),
        spectrum_label,
    )
    FISPACT_REFRENCE_SPECTRA[spectrum_label] = spctrum_datum_stored

# load group structures
with open(_FIPASCT_GROUPS_FILEPATH, "r", encoding="utf8") as input_file:
    GROUP_STRUCTURES = json.load(input_file)
GROUP_STRUCTURES = {int(k): v for k, v in GROUP_STRUCTURES.items()}

# load MT reaction lookup
with open(_MT_REACTIONS_FILEPATH, "r", encoding="utf8") as input_file:
    MT_REACTIONS = json.load(input_file)
MT_REACTIONS = {int(k): v for k, v in MT_REACTIONS.items()}
MT_NUMBERS = {v: k for k, v in MT_REACTIONS.items()}

# constants
BOLTZMANN = 1.380649e-23
BOLTZMANN_UNIT = "J K^-1"
NEUTRON_RAM = 1.00866491606
NEUTRON_RAM_UNIT = "g mol^-1"
GAS_CONSTANT = 8.314462618
GAS_CONTANT_UNIT = "J mol^-1 K^-1"
AVOGADRO = 6.02214076e23
AVOGADRO_UNIT = "mol^-1"
ELEMENTARY_CHARGE = 1.602176634e-19
ELEMENTARY_CHARGE_UNIT = "C"
