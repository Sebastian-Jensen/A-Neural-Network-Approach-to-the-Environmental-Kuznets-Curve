# ======================================================================================================================
"""
This script is used to reproduce plots for figures 8, 9, A16, A17.

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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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
else:
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

CO2 = pd.read_excel('Data/CO2_GCP.xlsx', sheet_name='Python', index_col=0) * 3.664
CO2.sort_index(axis=1, inplace=True)

CO2_cons = pd.read_excel('Data/CO2_consumption_based.xlsx', sheet_name='Python', index_col=0) * 3.664
CO2_cons.sort_index(axis=1, inplace=True)

where_CO2_cons = np.isnan(CO2_cons)

CO2_star = CO2[~where_CO2_cons]

# Squared residuals
if specification == 'static':
    filename = 'Squared residuals/resid_sq_' + str(specification) + '_regional_CO2_cons'

    infile = open(filename, 'rb')
    resid_sq = pickle.load(infile)
    infile.close()


#%% Constructing models
ghg_names = ['CO2_cons', 'CO2_star']

models = {}
nodes = {}
bic = {}

for ghg_name in ghg_names:
    # Preparing data
    gdp, ghg, pop = fPrepare(GDP, POP, DEF, PPP, eval(ghg_name))

    # Constructing model
    nodes[ghg_name], bic[ghg_name] = fNodes(specification, 'regional', ghg_name=ghg_name)

    tf.keras.backend.clear_session()
    models[ghg_name] = Model(nodes[ghg_name], gdp, ghg, pop_train=pop, formulation='regional')
    models[ghg_name].load_params('Model Parameters/' + specification.capitalize() + ' model/Regional/' + ghg_name + '/parameters_' + str(nodes[ghg_name]))
    models[ghg_name].in_sample_predictions()

# Preparing consumption data for plots
gdp, ghg, pop = fPrepare(GDP, POP, DEF, PPP, CO2_cons)

regions = list(gdp.keys())


# %% Creating model for the standard deviation
if specification == 'static':
    nodes_sd = (2,)

    tf.keras.backend.clear_session()

    model_sd = Model_sd(nodes_sd, gdp, resid_sq, pop, 'regional')
    model_sd.load_params('Model Parameters sd/' + specification.capitalize() + ' model/Regional/CO2_cons/parameters_' + str(nodes_sd))
    model_sd.in_sample_predictions()


#%% Loading benchmark estimates
benchmark_models = ['quadratic', 'cubic', 'splines_seg_10']

Benchmark_EKC = {}
Benchmark_time_FE = {}

for benchmark_model in benchmark_models:
    Benchmark_EKC[benchmark_model] = {}
    Benchmark_time_FE[benchmark_model] = {}

    for region in regions:
        Benchmark_EKC[benchmark_model][region] = pd.read_excel('Benchmark models/Results/CO2_cons/Regional_EKC_' + benchmark_model + '.xlsx', index_col=0, sheet_name=region)
        Benchmark_time_FE[benchmark_model][region] = pd.read_excel('Benchmark models/Results/CO2_cons/Regional_time_FE_' + benchmark_model + '.xlsx', index_col=0, sheet_name=region)


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
        # Functional relationship
        x_test = np.reshape(np.linspace(models['CO2_cons'].Min[region], models['CO2_cons'].Max[region], 10000), (-1, 1))
        y_test_cons = models['CO2_cons'].predict(x_test, region)
        y_test_cons_sd = model_sd.predict(x_test, region)
        y_test_star = models['CO2_star'].predict(x_test, region)

        plt.plot(y_test_cons.index.values, y_test_cons.values, color='black', label='CO2$^C$', linewidth=2)
        plt.plot(y_test_cons.index.values, y_test_cons.values + 1.96 * np.sqrt(y_test_cons_sd.values), color='black', label='95% PI for CO$_2$$^C$', linestyle='dotted', linewidth=2)
        plt.plot(y_test_cons.index.values, y_test_cons.values - 1.96 * np.sqrt(y_test_cons_sd.values), color='black', linestyle='dotted', linewidth=2)
        plt.plot(y_test_star.index.values, y_test_star.values, color='gray', linestyle='dashed', label='CO2$^\star$')
        plt.axvspan(models['CO2_cons'].quant05[region], models['CO2_cons'].quant95[region], facecolor='black', alpha=0.1, label='.05/.95 quantile')
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.xlabel('log(GDP)', fontsize=22)
        plt.ylabel('log(CO$_2$)', fontsize=22)
        plt.show()

        # Time fixed effects
        plt.plot(np.array(models['CO2_cons'].beta[region].index), np.reshape(np.array(models['CO2_cons'].beta[region]), (-1,)), color='black', label='CO2$^C$', linewidth=2)
        plt.plot(np.array(models['CO2_star'].beta[region].index), np.reshape(np.array(models['CO2_star'].beta[region]), (-1,)), color='gray', linestyle='dashed', label='CO2$^\star$')
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.show()

    # Benchmark results
    for region in regions:
        # Functional relationship
        plt.plot(Benchmark_EKC['quadratic'][region].index.values, Benchmark_EKC['quadratic'][region].values, linestyle=linestyles_dict['densely dashdotdotted'], color='black', label='Quadratic model')
        plt.plot(Benchmark_EKC['cubic'][region].index.values, Benchmark_EKC['cubic'][region].values, linestyle=linestyles_dict['densely dashdotted'], color='black', label='Cubic model')
        plt.plot(Benchmark_EKC['splines_seg_10'][region].index.values, Benchmark_EKC['splines_seg_10'][region].values, linestyle='dashed', color='black', label='Spline-based model')

        plt.axvspan(models['CO2_cons'].quant05[region], models['CO2_cons'].quant95[region], facecolor='black', alpha=0.1, label='.05/.95 quantile')
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.xlabel('log(GDP)', fontsize=22)
        plt.ylabel('log(CO$_2$)', fontsize=22)
        plt.show()

        # Time fixed effects
        plt.plot(Benchmark_time_FE['quadratic'][region].index.values, Benchmark_time_FE['quadratic'][region].values, linestyle=linestyles_dict['densely dashdotdotted'], color='black', label='Quadratic model')
        plt.plot(Benchmark_time_FE['cubic'][region].index.values, Benchmark_time_FE['cubic'][region].values, linestyle=linestyles_dict['densely dashdotted'], color='black', label='Cubic model')
        plt.plot(Benchmark_time_FE['splines_seg_10'][region].index.values, Benchmark_time_FE['splines_seg_10'][region].values, linestyle='dashed', color='black', label='Spline-based model')
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.show()

elif specification == 'dynamic':
    # Dynamic surfaces
    zlim = {}

    for ghg_name in ghg_names:
        for region in regions:
            ax1 = models['CO2_cons'].time_periods[models['CO2_cons'].time_periods_na[region]:]
            ax2 = np.linspace(models['CO2_cons'].Min[region].min(axis=0), models['CO2_cons'].Max[region].max(axis=0), 10000)

            ax1, ax2 = np.meshgrid(ax1, ax2)

            ax1_vec = np.reshape(ax1, (-1, 1), order='F')
            ax2_vec = np.reshape(ax2, (-1, 1), order='F')

            pred_df = models[ghg_name].predict(ax2_vec, region, ax1_vec)

            pred = np.reshape(np.array(pred_df['y']), (10000, models['CO2_cons'].T - models['CO2_cons'].time_periods_na[region]), order='F')

            pred_surf = pred.copy()

            vec_min = np.reshape(models['CO2_cons'].Min[region][models['CO2_cons'].time_periods_na[region]:].values, (1, -1), order='F')
            vec_max = np.reshape(models['CO2_cons'].Max[region][models['CO2_cons'].time_periods_na[region]:].values, (1, -1), order='F')
            vec_quantL = np.reshape(models['CO2_cons'].quant05[region][models['CO2_cons'].time_periods_na[region]:].values, (1, -1), order='F')
            vec_quantU = np.reshape(models['CO2_cons'].quant95[region][models['CO2_cons'].time_periods_na[region]:].values, (1, -1), order='F')

            ax3 = pred.copy()
            ax3[:, :] = 6
            ax3[np.where(ax2 < vec_quantL)] = 4.5
            ax3[np.where(ax2 > vec_quantU)] = 4.5

            ax3[np.where(ax2 < vec_min)] = 3
            ax3[np.where(ax2 > vec_max)] = 3

            minn, maxx = ax3.min(), ax3.max()
            norm = matplotlib.colors.Normalize(0, 6)
            m = plt.cm.ScalarMappable(norm=norm, cmap='Blues')

            m.set_array([])
            fcolors = m.to_rgba(ax3)

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(ax1, ax2, pred_surf, linewidth=0, antialiased=True, facecolors=fcolors, shade=True, alpha=0.8)
            ax.set_ylabel('              log(GDP)', fontsize=22)
            ax.set_zlabel('log(CO$_2$)', fontsize=22)
            ax.zaxis.set_tick_params(labelsize=22)
            ax.xaxis.set_tick_params(labelsize=22)
            ax.yaxis.set_tick_params(labelsize=22)
            ax.yaxis.labelpad = 15
            ax.zaxis.labelpad = 15
            if ghg_name == 'CO2_star':
                ax.set_zlim([zlim[region][0], zlim[region][1]])
            plt.title(region + '\n', fontsize=22)
            ax.view_init(52, -40)
            if region == 'OECD':
                ax.yaxis.set_ticks([2, 3, 4])
            if region == 'Asia':
                ax.yaxis.set_ticks([1, 2, 3, 4])
            if region == 'LAM':
                ax.yaxis.set_ticks([2, 3])

            plt.show()

            if ghg_name == 'CO2_cons':
                zlim[region] = ax.set_zlim()