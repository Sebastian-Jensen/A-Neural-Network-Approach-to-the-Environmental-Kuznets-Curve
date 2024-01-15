# ======================================================================================================================
"""
This script is used to estimate a static model for the standard deviation based on the squared residuals from a
model for the mean (computed in Estimate_models.py), then save the parameters.

OBS.: this script may take substantial time to finish. We ran this script in parallel on a server.

Required subroutines:
* fAlign.py
* fPrepare.py
* Static_NN_model_sd.py

By Sebastian Jensen
Jan, 2024
Aarhus University
CREATES
"""
# ======================================================================================================================

# Importing libraries
import os; os.environ['PYTHONHASHSEED'] = str(0)

import sys
sys.path.append('Functions')

import numpy as np
import tensorflow as tf
import pandas as pd
import random
import pickle

from fPrepare import fPrepare
from Static_NN_model_sd import static_model_sd as Model


# %% Setting choice parameters
no_inits = 3                      # number of different initializations
seed_value = -1                   # initial seed value minus one

ghg_name = 'CO2'                  # Must be 'CO2', 'CO2_cons', or 'CO2_star'
specification_mean = 'dynamic'    # Specification from which squared residuals were computed. Must be 'static' or 'dynamic'
formulation_sd = 'regional'       # Formulation to be used for modeling the standard deviation (should equal the formulation from which squared residuals were computed). Must be 'global', 'regional', or 'national'

nodes = (2,)                      # Tuple used to specify model architecture. Must be (x,), (x,y), or (x,y,z)


# %% Setting optional choice parameters
lr = 0.001                        # initial learning rate for the Adam optimizer
min_delta = 1e-5                  # tolerance to be used for optimization
patience = 100                    # patience to be used for optimization
verbose = False                   # verbosity mode for optimization


# %% Loading data
GDP = pd.read_excel('Data/GDP.xlsx', sheet_name='Python', index_col=0)
GDP.sort_index(axis=1, inplace=True)

POP = pd.read_excel('Data/Population.xlsx', sheet_name='Python', index_col=0) / 1e6
POP.sort_index(axis=1, inplace=True)

DEF = pd.read_excel('Data/Deflator.xlsx', sheet_name='Python', index_col=0)
DEF.sort_index(axis=1, inplace=True)

PPP = pd.read_excel('Data/PPP.xlsx', sheet_name='Python', index_col=0)
PPP.sort_index(axis=1, inplace=True)

if ghg_name == 'CO2':
    GHG = pd.read_excel('Data/CO2_GCP.xlsx', sheet_name='Python', index_col=0) * 3.664
    GHG.sort_index(axis=1, inplace=True)

elif ghg_name == 'CO2_cons':
    GHG = pd.read_excel('Data/CO2_consumption_based.xlsx', sheet_name='Python', index_col=0) * 3.664
    GHG.sort_index(axis=1, inplace=True)

elif ghg_name == 'CO2_star':
    CO2 = pd.read_excel('Data/CO2_GCP.xlsx', sheet_name='Python', index_col=0) * 3.664
    CO2.sort_index(axis=1, inplace=True)

    CO2_cons = pd.read_excel('Data/CO2_consumption_based.xlsx', sheet_name='Python', index_col=0) * 3.664
    CO2_cons.sort_index(axis=1, inplace=True)

    where_CO2_cons = np.isnan(CO2_cons)

    GHG = CO2[~where_CO2_cons]

# Squared residuals
filename = 'Squared residuals/resid_sq_' + specification_mean + '_' + formulation_sd + '_' + ghg_name

infile = open(filename, 'rb')
resid_sq = pickle.load(infile)
infile.close()


# %% Preparing data
gdp_est, ghg_est, pop_est = fPrepare(GDP, POP, DEF, PPP, GHG)


# %% Computing BIC
models_tmp = [None] * no_inits
BIC_tmp = [None] * no_inits

for j in range(no_inits):
    seed_value = seed_value + 1

    tf.keras.backend.clear_session()
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    models_tmp[j] = Model(nodes=nodes, x_train=gdp_est, y_train=resid_sq, pop_train=pop_est, formulation=formulation_sd)

    models_tmp[j].fit(lr=lr, min_delta=min_delta, patience=patience, verbose=verbose)

    models_tmp[j].in_sample_predictions()
    BIC_tmp[j] = models_tmp[j].BIC

where = np.where(BIC_tmp == np.min(BIC_tmp))[0][0]

# Saving parameters
models_tmp[where].save_params('Model Parameters sd/' + specification_mean.capitalize() + ' model/' + formulation_sd.capitalize() + '/' + ghg_name + '/parameters_' + str(nodes))


