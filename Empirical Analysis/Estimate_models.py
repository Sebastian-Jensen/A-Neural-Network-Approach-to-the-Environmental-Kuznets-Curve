# ======================================================================================================================
"""
This script is used to estimate and calculate the BIC for each model architecture described in the paper, then save the
parameters, BIC, and squared residuals.

OBS.: this script may take substantial time to finish. We ran this script in parallel on a server.

Required subroutines:
* fAlign.py
* fPrepare.py
* Static_NN_model.py
* Dynamic_NN_model.py

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


# %% Setting choice parameters
no_inits = 10                     # number of different initializations
seed_value = -1                   # initial seed value minus one

ghg_name = 'CO2'                  # Must be 'CO2', 'CO2_cons', or 'CO2_star'
specification = 'dynamic'         # Must be 'static' or 'dynamic'
formulation = 'regional'          # Must be 'global', 'regional', or 'national'


# %% Setting optional choice parameters
lr = 0.001                        # initial learning rate for the Adam optimizer
min_delta = 1e-6                  # tolerance to be used for optimization
patience = 100                    # patience to be used for optimization
verbose = False                   # verbosity mode for optimization


# %% Loading model class
if specification == 'static':
    from Static_NN_model import static_model as Model

elif specification == 'dynamic':
    from Dynamic_NN_model import dynamic_model as Model


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


# %% Preparing data
gdp_est, ghg_est, pop_est = fPrepare(GDP, POP, DEF, PPP, GHG)


# %% Creating list of architectures to consider
nodes_list = [(2,), (4,), (8,), (16,), (32,), (2,2,), (4,2,), (4,4,), (8,2,), (8,4,), (8,8,), (16,2,), (16,4,), (16,8,),
              (16,16,), (32,2,), (32,4,), (32,8,), (32,16,), (32,32,), (2,2,2,), (4,2,2,), (4,4,2,), (4,4,4,), (8,2,2,),
              (8,4,2,), (8,4,4,), (8,8,2), (8,8,4), (8,8,8,), (16,2,2,), (16,4,2,), (16,4,4,), (16,8,2,), (16,8,4,),
              (16,8,8,), (16,16,2,), (16,16,4,), (16,16,8,), (16,16,16,), (32,2,2,), (32,4,2,), (32,4,4,), (32,8,2,),
              (32,8,4,), (32,8,8,), (32,16,2,), (32,16,4,), (32,16,8,), (32,16,16,), (32,32,2,), (32,32,4,), (32,32,8,),
              (32,32,16,), (32,32,32,)]


# %% Computing BIC
BIC = [None] * len(nodes_list)
models = [None] * len(nodes_list)

for i in range(len(nodes_list)):
    models_tmp = [None] * no_inits
    BIC_tmp = [None] * no_inits

    seed_value_tmp = seed_value

    for j in range(no_inits):
        seed_value_tmp = seed_value_tmp + 1

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed_value_tmp)
        np.random.seed(seed_value_tmp)
        random.seed(seed_value_tmp)

        models_tmp[j] = Model(nodes=nodes_list[i], x_train=gdp_est, y_train=ghg_est, pop_train=pop_est, formulation=formulation)

        models_tmp[j].fit(lr=lr, min_delta=min_delta, patience=patience, verbose=verbose)

        models_tmp[j].in_sample_predictions()
        BIC_tmp[j] = models_tmp[j].BIC

    where = np.where(BIC_tmp == np.min(BIC_tmp))[0][0]

    # Storing BIC and model from best initialization
    BIC[i] = BIC_tmp[where]
    models[i] = models_tmp[where]

    # Saving BIC from best initialization for each architecture
    np.save('BIC/' + specification.capitalize() + ' model/' + formulation.capitalize() + '/' + ghg_name + '/BIC_' + str(nodes_list[i]), np.array(models_tmp[where].BIC))

    # Saving parameters from best initialization for each architecture
    models_tmp[where].save_params('Model Parameters/' + specification.capitalize() + ' model/' + formulation.capitalize() + '/' + ghg_name + '/parameters_' + str(nodes_list[i]))

# Saving squared residuals from model selected by BIC (for creating prediction intervals)
where = np.where(BIC == np.min(BIC))[0][0]

regions = list(gdp_est.keys())
resid_sq = {}

for region in regions:
    resid_sq[region] = (models[where].in_sample_pred[region] - models[where].y_train_df[region])**2

outfile = open('Squared residuals/resid_sq_' + specification + '_' + formulation + '_' + ghg_name, 'wb')
pickle.dump(resid_sq, outfile)
outfile.close()
