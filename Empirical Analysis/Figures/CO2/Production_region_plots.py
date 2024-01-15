# ======================================================================================================================
"""
This script is used to reproduce plots for figures 4, 6, 7.

Required subroutines:
* fPrepare.py
* fAlign.py
* fNodes.py
* Static_NN_model.py
* Static_NN_model_sd.py
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

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns; sns.set_style('darkgrid')

from fPrepare import fPrepare
from fNodes import fNodes
from Static_NN_model_sd import static_model_sd as Model_sd

import pickle

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


# %% Setting choice parameters
specification = 'dynamic'  # Must be 'static' or 'dynamic'


# %% Loading model class
if specification == 'static':
    from Static_NN_model import static_model as Model

elif specification == 'dynamic':
    from Dynamic_NN_model import dynamic_model as Model


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
filename = 'Squared residuals/resid_sq_' + str(specification) + '_regional_CO2'

infile = open(filename, 'rb')
resid_sq = pickle.load(infile)
infile.close()


#%%  Preparing data
gdp, ghg, pop = fPrepare(GDP, POP, DEF, PPP, GHG)

regions = list(gdp.keys())


#%% Loading benchmark estimates
benchmark_models = ['quadratic', 'cubic', 'splines_seg_10']

Benchmark_EKC = {}
Benchmark_time_FE = {}

for benchmark_model in benchmark_models:
    Benchmark_EKC[benchmark_model] = {}
    Benchmark_time_FE[benchmark_model] = {}

    for region in regions:
        Benchmark_EKC[benchmark_model][region] = pd.read_excel('Benchmark models/Results/CO2/Regional_EKC_' + benchmark_model + '.xlsx', index_col=0, sheet_name=region)
        Benchmark_time_FE[benchmark_model][region] = pd.read_excel('Benchmark models/Results/CO2/Regional_time_FE_' + benchmark_model + '.xlsx', index_col=0, sheet_name=region)


#%% Constructing model
nodes, bic = fNodes(specification, 'regional', ghg_name='CO2')

tf.keras.backend.clear_session()
model = Model(nodes, gdp, ghg, pop_train=pop, formulation='regional')
model.load_params('Model Parameters/' + specification.capitalize() + ' model/Regional/CO2/parameters_' + str(nodes))
model.in_sample_predictions()


# %% Creating model for the standard deviation
nodes_sd = (2,)

tf.keras.backend.clear_session()

model_sd = Model_sd(nodes_sd, gdp, resid_sq, pop, 'regional')
model_sd.load_params('Model Parameters sd/' + specification.capitalize() + ' model/Regional/CO2/parameters_' + str(nodes_sd))
model_sd.in_sample_predictions()


#%% Ploting
for region in regions:
    # Scatter plots
    plt.scatter(np.log(gdp[region] / pop[region]), np.log(ghg[region] / pop[region]), s=2, alpha=0.5, color='black')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel('log(GDP)', fontsize=22)
    plt.ylabel(region + '\n\n' + 'log(CO$_2$)', fontsize=22)
    plt.show()

if specification == 'static':
    for region in regions:
        # Function with ben
        x_test = np.reshape(np.linspace(model.Min[region], model.Max[region], 10000), (-1, 1))
        y_test = model.predict(x_test, region)
        y_test_sd = model_sd.predict(x_test, region)

        plt.plot(Benchmark_EKC['quadratic'][region].index.values, Benchmark_EKC['quadratic'][region].values, linestyle=linestyles_dict['densely dashdotdotted'], color='gray', label='Quadratic model')
        plt.plot(Benchmark_EKC['cubic'][region].index.values, Benchmark_EKC['cubic'][region].values, linestyle=linestyles_dict['densely dashdotted'], color='gray', label='Cubic model')
        plt.plot(Benchmark_EKC['splines_seg_10'][region].index.values, Benchmark_EKC['splines_seg_10'][region].values, linestyle='dashed', color='gray', label='Spline-based model')

        plt.plot(y_test.index.values, y_test.values, color='black', linewidth=2)
        plt.plot(y_test.index.values, y_test.values + 1.96 * np.sqrt(y_test_sd.values), color='black', label='95% PI for Static NN', linestyle='dotted', linewidth=2)
        plt.plot(y_test.index.values, y_test.values - 1.96 * np.sqrt(y_test_sd.values), color='black', linestyle='dotted', linewidth=2)

        plt.axvspan(model.quant05[region], model.quant95[region], facecolor='black', alpha=0.1, label='.05/.95 quantile')
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.xlabel('log(GDP)', fontsize=22)
        plt.ylabel('log(CO$_2$)', fontsize=22)
        plt.show()

        # Time fixed effects with ben
        plt.plot(Benchmark_time_FE['quadratic'][region].index.values, Benchmark_time_FE['quadratic'][region].values, linestyle=linestyles_dict['densely dashdotdotted'], color='gray', label='Quadratic model')
        plt.plot(Benchmark_time_FE['cubic'][region].index.values, Benchmark_time_FE['cubic'][region].values, linestyle=linestyles_dict['densely dashdotted'], color='gray', label='Cubic model')
        plt.plot(Benchmark_time_FE['splines_seg_10'][region].index.values, Benchmark_time_FE['splines_seg_10'][region].values, linestyle='dashed', color='gray', label='Spline-based model')

        plt.plot(np.array(model.beta[region].index), np.reshape(np.array(model.beta[region]), (-1,)), color='black', linewidth=2)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.show()

elif specification == 'dynamic':
    # Slices of dynamic surfaces
    region = 'OECD'
    years = [1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2018]

    for year in years:
        x_test = np.linspace(model.Min[region][year], model.Max[region][year], 10000)
        y_test_sd = model_sd.predict(x_test, region)

        ax1 = model.time_periods[model.time_periods_na[region]:]
        ax2 = np.linspace(model.Min[region][year], model.Max[region][year], 10000)

        ax1, ax2 = np.meshgrid(ax1, ax2)

        ax1_vec = np.reshape(ax1, (-1, 1), order='F')
        ax2_vec = np.reshape(ax2, (-1, 1), order='F')

        pred_df = model.predict(ax2_vec, region, ax1_vec)

        pred = np.reshape(np.array(pred_df['y']), (10000, model.T - model.time_periods_na[region]), order='F')

        plt.axvspan(model.quant05[region][year], model.quant95[region][year], facecolor='black', alpha=0.1, label='.05/.95 quantile')

        year_adj = year - 1960
        plt.plot(ax2[:, year_adj], pred[:, year_adj], color='black', label='Estimated function')
        plt.plot(ax2[:, year_adj], pred[:, year_adj] + 1.96 * np.reshape(np.sqrt(y_test_sd.values), (-1,)),
                 color='black', linestyle='dotted', label='95% PI')
        plt.plot(ax2[:, year_adj], pred[:, year_adj] - 1.96 * np.reshape(np.sqrt(y_test_sd.values), (-1,)),
                 color='black', linestyle='dotted')

        if year == years[0]:
            plt.legend(loc='upper left', fancybox=True, shadow=False, ncol=1, markerscale=1, prop={'size': 16})

        plt.title(year, fontsize=22)
        plt.xlabel('log(GDP)', fontsize=22)
        plt.ylabel('log(CO$_2$)', fontsize=22)
        plt.yticks(fontsize=22)
        plt.xticks(fontsize=22)
        plt.xlim(0.0, 4.4)

        plt.show()

    # Dynamic surfaces
    ax1 = {}
    ax2 = {}
    ax3 = {}
    pred_surf = {}

    for region in ['OECD', 'REF', 'Asia', 'MAF', 'LAM']:
        ax1[region] = model.time_periods[model.time_periods_na[region]:]
        ax2[region] = np.linspace(model.Min[region].min(axis=0), model.Max[region].max(axis=0), 10000)

        ax1[region], ax2[region] = np.meshgrid(ax1[region], ax2[region])

        ax1_vec = np.reshape(ax1[region], (-1, 1), order='F')
        ax2_vec = np.reshape(ax2[region], (-1, 1), order='F')

        pred_df = model.predict(ax2_vec, region, ax1_vec)

        pred = np.reshape(np.array(pred_df['y']), (10000, model.T - model.time_periods_na[region]), order='F')

        pred_surf[region] = pred.copy()

        vec_min = np.reshape(model.Min[region][model.time_periods_na[region]:].values, (1, -1), order='F')
        vec_max = np.reshape(model.Max[region][model.time_periods_na[region]:].values, (1, -1), order='F')
        vec_quantL = np.reshape(model.quant05[region][model.time_periods_na[region]:].values, (1, -1), order='F')
        vec_quantU = np.reshape(model.quant95[region][model.time_periods_na[region]:].values, (1, -1), order='F')

        ax3[region] = pred.copy()
        ax3[region][:, :] = 6
        ax3[region][np.where(ax2[region] < vec_quantL)] = 4.5
        ax3[region][np.where(ax2[region] > vec_quantU)] = 4.5

        ax3[region][np.where(ax2[region] < vec_min)] = 3
        ax3[region][np.where(ax2[region] > vec_max)] = 3

        minn, maxx = ax3[region].min(), ax3[region].max()
        norm = matplotlib.colors.Normalize(0, 6)
        m = plt.cm.ScalarMappable(norm=norm, cmap='Blues')
        m.set_array([])
        fcolors = m.to_rgba(ax3[region])

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(ax1[region], ax2[region], pred_surf[region], linewidth=0, antialiased=True, facecolors=fcolors, shade=True, alpha=0.8)
        ax.set_ylabel('              log(GDP)', fontsize=22)
        lgd = ax.set_zlabel('log(CO$_2$)', fontsize=22)
        ax.zaxis.set_tick_params(labelsize=22)
        ax.xaxis.set_tick_params(labelsize=22)
        if region == 'REF':
            ax.xaxis.set_ticks([1990, 2000, 2010])
        else:
            ax.xaxis.set_ticks([1960, 1980, 2000])
        ax.yaxis.set_tick_params(labelsize=22)
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 15
        plt.title(region + '\n', fontsize=22)
        ax.view_init(52, -40)
        if region == 'Asia':
            ax.set_zlim(-3, 4)
        if region == 'LAM':
            ax.set_zlim(-0.5, 3.5)
        if region == 'OECD':
            ax.yaxis.set_ticks([1, 2, 3, 4])
        if region == 'REF':
            ax.yaxis.set_ticks([1, 2, 3])

        plt.show()
