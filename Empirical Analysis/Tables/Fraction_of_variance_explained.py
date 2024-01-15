# ======================================================================================================================
"""
This script is used to reproduce fractions of variance explained for Table 3.

Required subroutines:
* fPrepare.py
* fAlign.py
* fNodes.py
* fDummies
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
import sys; sys.path.append('Functions')

import numpy as np
import tensorflow as tf
import pandas as pd

from numpy.linalg import inv as inv

from fPrepare import fPrepare
from fDummies import fDummies
from fNodes import fNodes


# %% Setting choice parameters
specification = 'static'  # Must be 'static' or 'dynamic'
formulation = 'regional'  # Must be 'global', 'regional', or 'national'


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

GHG = pd.read_excel('Data/CO2_GCP.xlsx', sheet_name='Python', index_col=0) * 3.664
GHG.sort_index(axis=1, inplace=True)


# %%  Preparing data
gdp, ghg, pop = fPrepare(GDP, POP, DEF, PPP, GHG)

regions = list(gdp.keys())


# %% Constructing model
nodes, bic = fNodes(specification, formulation, ghg_name='CO2')

tf.keras.backend.clear_session()
model = Model(nodes, gdp, ghg, pop_train=pop, formulation=formulation)
model.load_params('Model Parameters/' + specification.capitalize() + ' model/' + formulation.capitalize() + '/CO2/parameters_' + str(nodes))
model.in_sample_predictions()


# %% Creating concatenated data
for region in regions:
    if region == regions[0]:
        gdp_concat = gdp[regions[0]].copy()
        ghg_concat = ghg[regions[0]].copy()
        pop_concat = pop[regions[0]].copy()

    else:
        gdp_concat = pd.concat([gdp_concat, gdp[region]], axis=1)
        ghg_concat = pd.concat([ghg_concat, ghg[region]], axis=1)
        pop_concat = pd.concat([pop_concat, pop[region]], axis=1)


# %% Performing OLS regression and calculating R2 from country fixed effects
R2_country_FE = {}

# Global
X = np.log(gdp_concat / pop_concat)
Y = np.log(ghg_concat / pop_concat)

Delta_1, _ = fDummies(X)

where_mat = X.isna().values.T
y = np.reshape(Y.values[~where_mat.T], (-1, 1), order='F')

B = inv(Delta_1.T @ Delta_1) @ Delta_1.T @ y
XB = Delta_1 @ B

SSR = np.sum((y - XB) ** 2)
SST = np.sum((y - np.mean(y)) ** 2)
R2_country_FE['global'] = 1 - SSR / SST

# regional
for region in regions:
    X = np.log(gdp[region] / pop[region])
    Y = np.log(ghg[region] / pop[region])

    Delta_1, _ = fDummies(X)

    time_periods_not_na = np.sum(~np.isnan(X), axis=1) > 0
    time_periods_na = np.sum(~time_periods_not_na)

    where_mat = X.isna().values.T

    y = np.reshape(Y.values[~where_mat.T], (-1, 1), order='F')

    B = inv(Delta_1.T @ Delta_1) @ Delta_1.T @ y
    XB = Delta_1 @ B

    SSR = np.sum((y - XB) ** 2)
    SST = np.sum((y - np.mean(y)) ** 2)
    R2_country_FE[region] = 1 - SSR / SST


# %% Performing OLS regression and calculating R2 from time fixed effects
R2_time_FE_global = {}
R2_time_FE_region = {}

# Global
X = np.log(gdp_concat / pop_concat)
Y = np.log(ghg_concat / pop_concat)

_, Delta_2 = fDummies(X)

where_mat = X.isna().values.T
y = np.reshape(Y.values[~where_mat.T], (-1, 1), order='F')

B = inv(Delta_2.T @ Delta_2) @ Delta_2.T @ y
XB = Delta_2 @ B

SSR = np.sum((y - XB) ** 2)
SST = np.sum((y - np.mean(y)) ** 2)
R2_time_FE_global['global'] = 1 - SSR / SST

# Global for regions
for region in regions:
    Y = np.log(ghg[region] / pop[region])

    where_mat_tmp = (~gdp[region].isna()).values

    XB_tmp = np.reshape((where_mat_tmp * B)[where_mat_tmp], (-1, 1), order='F')
    y = np.reshape(Y.values[where_mat_tmp], (-1, 1), order='F')

    SSR = np.sum((y - XB_tmp) ** 2)
    SST = np.sum((y - np.mean(y)) ** 2)
    R2_time_FE_global[region] = 1 - SSR / SST

# regional
for region in regions:
    X = np.log(gdp[region] / pop[region])
    Y = np.log(ghg[region] / pop[region])

    _, Delta_2 = fDummies(X)

    time_periods_not_na = np.sum(~np.isnan(X), axis=1) > 0
    time_periods_na = np.sum(~time_periods_not_na)

    Delta_2 = Delta_2[:, time_periods_na:]

    where_mat = X.isna().values.T

    y = np.reshape(Y.values[~where_mat.T], (-1, 1), order='F')

    B = inv(Delta_2.T @ Delta_2) @ Delta_2.T @ y
    XB = Delta_2 @ B

    SSR = np.sum((y - XB) ** 2)
    SST = np.sum((y - np.mean(y)) ** 2)
    R2_time_FE_region[region] = 1 - SSR / SST

    if region == regions[0]:
        XB_stack = XB
        y_stack = y
    else:
        XB_stack = np.vstack((XB_stack, XB))
        y_stack = np.vstack((y_stack, y))

# regional for global
SSR = np.sum((y_stack - XB_stack) ** 2)
SST = np.sum((y_stack - np.mean(y_stack)) ** 2)
R2_time_FE_region['global'] = 1 - SSR / SST


# %% Printing results
print('R2', '\n')
print(specification.capitalize(), formulation, ':', model.R2, '\n')
print('Country effects:', R2_country_FE, '\n')
print('Time effects global:', R2_time_FE_global, '\n')
print('Time effects regional:', R2_time_FE_region)