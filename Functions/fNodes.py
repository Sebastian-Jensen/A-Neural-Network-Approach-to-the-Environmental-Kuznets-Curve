# ======================================================================================================================
"""
This script creates a function used to return the neural network architecture selected by the BIC (computed in
Estimate_models.py).

By Sebastian Jensen
Jan, 2024
Aarhus University
CREATES
"""
# ======================================================================================================================

# Importing libraries
import numpy as np


def fNodes(specification, formulation, ghg_name):
    """
    ARGUMENTS
        * specification: str determining the specification of the model. Must be 'static' or 'dynamic'.
        * formulation:   str determining the formulation of the model. Must be one of 'global' or 'regional' or 'national'.
        * ghg_name:      str determining the dependent variable of the analysis. Must be 'CO2', 'CO2_cons', or 'CO2_star'.

    Returns
        * nodes: neural network architecture selected by BIC
        * bic:   BIC for selected neural network architecture
    """

    nodes_list = [(2,), (4,), (8,), (16,), (32,), (2, 2,), (4, 2,), (4, 4,), (8, 2,), (8, 4,), (8, 8,), (16, 2,),
                  (16, 4,), (16, 8,), (16, 16,), (32, 2,), (32, 4,), (32, 8,), (32, 16,), (32, 32,), (2, 2, 2,),
                  (4, 2, 2,), (4, 4, 2,), (4, 4, 4,), (8, 2, 2,), (8, 4, 2,), (8, 4, 4,), (8, 8, 2), (8, 8, 4),
                  (8, 8, 8,), (16, 2, 2,), (16, 4, 2,), (16, 4, 4,), (16, 8, 2,), (16, 8, 4,), (16, 8, 8,),
                  (16, 16, 2,), (16, 16, 4,), (16, 16, 8,), (16, 16, 16,), (32, 2, 2,), (32, 4, 2,), (32, 4, 4,),
                  (32, 8, 2,), (32, 8, 4,), (32, 8, 8,), (32, 16, 2,), (32, 16, 4,), (32, 16, 8,), (32, 16, 16,),
                  (32, 32, 2,), (32, 32, 4,), (32, 32, 8,), (32, 32, 16,), (32, 32, 32,)]

    BIC = [None] * len(nodes_list)

    for i in range(len(nodes_list)):
        BIC[i] = np.load('BIC/' + specification.capitalize() + ' model/' + formulation.capitalize() + '/' + ghg_name + '/BIC_' + str(nodes_list[i]) + '.npy')

    return nodes_list[np.where(BIC == np.min(BIC))[0][0]], np.min(BIC)
