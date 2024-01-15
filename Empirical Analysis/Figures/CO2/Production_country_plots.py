# ======================================================================================================================
"""
This script is used to reproduce plots for figures 5.

Required subroutines:
* fPrepare.py
* fAlign.py
* fNodes.py
* Static_NN_model.py
* Static_NN_model_sd.py

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

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('darkgrid')

import pickle

from fPrepare import fPrepare
from fNodes import fNodes
from Static_NN_model import static_model as Model
from Static_NN_model_sd import static_model_sd as Model_sd

from collections import OrderedDict

linestyles_dict = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])


# %% Loading data
ghg_name = 'CO2'

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

# Squared residuals
filename = 'Squared residuals/resid_sq_static_national_CO2'

infile = open(filename, 'rb')
resid_sq = pickle.load(infile)
infile.close()


# %%  Preparing data
gdp, ghg, pop = fPrepare(GDP, POP, DEF, PPP, GHG)

regions = list(gdp.keys())


# %% Loading benchmark estimates
benchmark_models = ['quadratic', 'cubic', 'splines_seg_10']

Benchmark_EKC = {}

for benchmark_model in benchmark_models:
    Benchmark_EKC[benchmark_model] = {}

    for region in regions:
        if region == 'OECD':
            for country in ['USA', 'DEU']:
                Benchmark_EKC[benchmark_model][country] = pd.read_excel('Benchmark models/Results/CO2/National_EKC_' + benchmark_model + '.xlsx', index_col=0, sheet_name=country)

        if region == 'REF':
            for country in ['RUS']:
                Benchmark_EKC[benchmark_model][country] = pd.read_excel('Benchmark models/Results/CO2/National_EKC_' + benchmark_model + '.xlsx', index_col=0, sheet_name=country)

        if region == 'Asia':
            for country in ['CHN', 'IND', 'JPN', 'KOR', 'IRN']:
                Benchmark_EKC[benchmark_model][country] = pd.read_excel('Benchmark models/Results/CO2/National_EKC_' + benchmark_model + '.xlsx', index_col=0, sheet_name=country)

        if region == 'MAF':
            for country in ['SAU']:
                Benchmark_EKC[benchmark_model][country] = pd.read_excel('Benchmark models/Results/CO2/National_EKC_' + benchmark_model + '.xlsx', index_col=0, sheet_name=country)


# %% Constructing model
nodes, bic = fNodes('static', 'national', ghg_name='CO2')

tf.keras.backend.clear_session()
model = Model(nodes, gdp, ghg, pop_train=pop, formulation='national')
model.load_params('Model Parameters/Static model/National/CO2/parameters_' + str(nodes))
model.in_sample_predictions()


# %% Creating model for the standard deviation
nodes_sd = (2,)

tf.keras.backend.clear_session()

model_sd = Model_sd(nodes_sd, gdp, resid_sq, pop, 'national')
model_sd.load_params('Model Parameters sd/Static model/National/CO2/parameters_' + str(nodes_sd))
model_sd.in_sample_predictions()


#%% Ploting
x_tmp = {}
y_tmp = {}
x_tmp_vec = {}
y_tmp_vec = {}
y_tmp_vec_sd = {}

for region in regions:
    for country in model.individuals[region]:
        if country in ['USA', 'DEU', 'RUS', 'CHN', 'IND', 'JPN', 'IRN', 'KOR', 'SAU']:
            x_tmp[country] = np.log(gdp[region] / pop[region])[country]
            y_tmp[country] = np.log(ghg[region] / pop[region])[country]

            x_tmp_vec[country] = np.reshape(np.linspace(x_tmp[country].min(), x_tmp[country].max(), 10000), (-1,1))

            fig = plt.figure()

            plt.scatter(x_tmp[country], y_tmp[country], label='Observations')

            y_tmp_vec[country] = model.predict(x_tmp_vec[country], idx=country)
            y_tmp_vec_sd[country] = model_sd.predict(x_tmp_vec[country], idx=country)

            plt.axvspan(model.quant05[country], model.quant95[country], facecolor='black', alpha=0.1, label='.05/.95 quantile')
            plt.xlabel('log(GDP)', fontsize=22)
            plt.ylabel('log(CO$_2$)', fontsize=22)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.title(country + str(np.where(model.individuals[region] == country)[0][0]), fontsize=22)

            if country in ['CHN']:
                plt.plot(Benchmark_EKC['quadratic'][country].index.values, Benchmark_EKC['quadratic'][country].values, linestyle=linestyles_dict['densely dashdotdotted'], color='gray', label='Quadratic model')
                plt.plot(Benchmark_EKC['cubic'][country].index.values, Benchmark_EKC['cubic'][country].values, linestyle=linestyles_dict['densely dashdotted'], color='gray', label='Cubic model')
                plt.plot(Benchmark_EKC['splines_seg_10'][country].index.values, Benchmark_EKC['splines_seg_10'][country].values, linestyle='dashed', color='gray', label='Spline-based model')

                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values, color='black', label='Static NN model', linewidth=2)
                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values + 1.96 * np.sqrt(y_tmp_vec_sd[country].values), color='black', label='95% PI for Static NN', linestyle='dotted', linewidth=2)
                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values - 1.96 * np.sqrt(y_tmp_vec_sd[country].values), color='black', linestyle='dotted', linewidth=2)

                plt.title('China', fontsize=22)
                plt.legend(loc='lower right', fancybox=True, shadow=False, ncol=1, markerscale=1, prop={'size': 12})

            if country in ['USA']:
                plt.plot(Benchmark_EKC['quadratic'][country].index.values, Benchmark_EKC['quadratic'][country].values, linestyle=linestyles_dict['densely dashdotdotted'], color='gray', label='Quadratic model')
                plt.plot(Benchmark_EKC['cubic'][country].index.values, Benchmark_EKC['cubic'][country].values, linestyle=linestyles_dict['densely dashdotted'], color='gray', label='Cubic model')
                plt.plot(Benchmark_EKC['splines_seg_10'][country].index.values, Benchmark_EKC['splines_seg_10'][country].values, linestyle='dashed', color='gray', label='Spline-based model')

                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values, color='black', label='Estimated function', linewidth=2)
                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values + 1.96 * np.sqrt(y_tmp_vec_sd[country].values), color='black', label='95% PI for Static NN', linestyle='dotted', linewidth=2)
                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values - 1.96 * np.sqrt(y_tmp_vec_sd[country].values), color='black', linestyle='dotted', linewidth=2)

                plt.title('United States', fontsize=22)

            if country in ['IND']:
                plt.plot(Benchmark_EKC['quadratic'][country].index.values, Benchmark_EKC['quadratic'][country].values, linestyle=linestyles_dict['densely dashdotdotted'], color='gray', label='Quadratic model')
                plt.plot(Benchmark_EKC['cubic'][country].index.values, Benchmark_EKC['cubic'][country].values, linestyle=linestyles_dict['densely dashdotted'], color='gray', label='Cubic model')
                plt.plot(Benchmark_EKC['splines_seg_10'][country].index.values, Benchmark_EKC['splines_seg_10'][country].values, linestyle='dashed', color='gray', label='Spline-based model')

                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values, color='black', label='Estimated function', linewidth=2)
                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values + 1.96 * np.sqrt(y_tmp_vec_sd[country].values), color='black', label='95% PI for Static NN', linestyle='dotted', linewidth=2)
                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values - 1.96 * np.sqrt(y_tmp_vec_sd[country].values), color='black', linestyle='dotted', linewidth=2)

                plt.title('India', fontsize=22)

            if country in ['RUS']:
                plt.plot(Benchmark_EKC['quadratic'][country].index.values, Benchmark_EKC['quadratic'][country].values, linestyle=linestyles_dict['densely dashdotdotted'], color='gray', label='Quadratic model')
                plt.plot(Benchmark_EKC['cubic'][country].index.values, Benchmark_EKC['cubic'][country].values, linestyle=linestyles_dict['densely dashdotted'], color='gray', label='Cubic model')
                plt.plot(Benchmark_EKC['splines_seg_10'][country].index.values, Benchmark_EKC['splines_seg_10'][country].values, linestyle='dashed', color='gray', label='Spline-based model')

                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values, color='black', label='Estimated function', linewidth=2)
                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values + 1.96 * np.sqrt(y_tmp_vec_sd[country].values), color='black', label='95% PI for Static NN', linestyle='dotted', linewidth=2)
                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values - 1.96 * np.sqrt(y_tmp_vec_sd[country].values), color='black', linestyle='dotted', linewidth=2)

                plt.title('Russia', fontsize=22)

            if country in ['JPN']:
                plt.plot(Benchmark_EKC['quadratic'][country].index.values, Benchmark_EKC['quadratic'][country].values, linestyle=linestyles_dict['densely dashdotdotted'], color='gray', label='Quadratic model')
                plt.plot(Benchmark_EKC['cubic'][country].index.values, Benchmark_EKC['cubic'][country].values, linestyle=linestyles_dict['densely dashdotted'], color='gray', label='Cubic model')
                plt.plot(Benchmark_EKC['splines_seg_10'][country].index.values, Benchmark_EKC['splines_seg_10'][country].values, linestyle='dashed', color='gray', label='Spline-based model')

                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values, color='black', label='Estimated function', linewidth=2)
                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values + 1.96 * np.sqrt(y_tmp_vec_sd[country].values), color='black', label='95% PI for Static NN', linestyle='dotted', linewidth=2)
                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values - 1.96 * np.sqrt(y_tmp_vec_sd[country].values), color='black', linestyle='dotted', linewidth=2)

                plt.title('Japan', fontsize=22)

            if country in ['DEU']:
                plt.plot(Benchmark_EKC['quadratic'][country].index.values, Benchmark_EKC['quadratic'][country].values, linestyle=linestyles_dict['densely dashdotdotted'], color='gray', label='Quadratic model')
                plt.plot(Benchmark_EKC['cubic'][country].index.values, Benchmark_EKC['cubic'][country].values, linestyle=linestyles_dict['densely dashdotted'], color='gray', label='Cubic model')
                plt.plot(Benchmark_EKC['splines_seg_10'][country].index.values, Benchmark_EKC['splines_seg_10'][country].values, linestyle='dashed', color='gray', label='Spline-based model')

                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values, color='black', label='Estimated function', linewidth=2)
                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values + 1.96 * np.sqrt(y_tmp_vec_sd[country].values), color='black', label='95% PI for Static NN', linestyle='dotted', linewidth=2)
                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values - 1.96 * np.sqrt(y_tmp_vec_sd[country].values), color='black', linestyle='dotted', linewidth=2)

                plt.title('Germany', fontsize=22)

            if country in ['IRN']:
                plt.plot(Benchmark_EKC['quadratic'][country].index.values, Benchmark_EKC['quadratic'][country].values, linestyle=linestyles_dict['densely dashdotdotted'], color='gray', label='Quadratic model')
                plt.plot(Benchmark_EKC['cubic'][country].index.values, Benchmark_EKC['cubic'][country].values, linestyle=linestyles_dict['densely dashdotted'], color='gray', label='Cubic model')
                plt.plot(Benchmark_EKC['splines_seg_10'][country].index.values, Benchmark_EKC['splines_seg_10'][country].values, linestyle='dashed', color='gray', label='Spline-based model')

                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values, color='black', label='Estimated function', linewidth=2)
                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values + 1.96 * np.sqrt(y_tmp_vec_sd[country].values), color='black', label='95% PI for Static NN', linestyle='dotted', linewidth=2)
                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values - 1.96 * np.sqrt(y_tmp_vec_sd[country].values), color='black', linestyle='dotted', linewidth=2)

                plt.title('Iran', fontsize=22)

            if country in ['KOR']:
                plt.plot(Benchmark_EKC['quadratic'][country].index.values, Benchmark_EKC['quadratic'][country].values,  linestyle=linestyles_dict['densely dashdotdotted'], color='gray', label='Quadratic model')
                plt.plot(Benchmark_EKC['cubic'][country].index.values, Benchmark_EKC['cubic'][country].values, linestyle=linestyles_dict['densely dashdotted'], color='gray', label='Cubic model')
                plt.plot(Benchmark_EKC['splines_seg_10'][country].index.values, Benchmark_EKC['splines_seg_10'][country].values, linestyle='dashed', color='gray', label='Spline-based model')

                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values, color='black', label='Estimated function', linewidth=2)
                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values + 1.96 * np.sqrt(y_tmp_vec_sd[country].values), color='black', label='95% PI for Static NN', linestyle='dotted', linewidth=2)
                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values - 1.96 * np.sqrt(y_tmp_vec_sd[country].values), color='black', linestyle='dotted', linewidth=2)

                plt.title('South Korea', fontsize=22)

            if country in ['SAU']:
                plt.plot(Benchmark_EKC['quadratic'][country].index.values, Benchmark_EKC['quadratic'][country].values, linestyle=linestyles_dict['densely dashdotdotted'], color='gray', label='Quadratic model')
                plt.plot(Benchmark_EKC['cubic'][country].index.values, Benchmark_EKC['cubic'][country].values, linestyle=linestyles_dict['densely dashdotted'], color='gray', label='Cubic model')
                plt.plot(Benchmark_EKC['splines_seg_10'][country].index.values, Benchmark_EKC['splines_seg_10'][country].values, linestyle='dashed', color='gray', label='Spline-based model')

                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values, color='black', label='Estimated function', linewidth=2)
                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values + 1.96 * np.sqrt(y_tmp_vec_sd[country].values), color='black', label='95% PI for Static NN', linestyle='dotted', linewidth=2)
                plt.plot(x_tmp_vec[country], y_tmp_vec[country].values - 1.96 * np.sqrt(y_tmp_vec_sd[country].values), color='black', linestyle='dotted', linewidth=2)

                plt.title('Saudi Arabia', fontsize=22)

            plt.show()
