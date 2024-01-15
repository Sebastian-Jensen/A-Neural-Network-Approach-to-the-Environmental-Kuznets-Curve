# ======================================================================================================================
"""
This script creates a function used to construct dummies.

By Sebastian Jensen
Jan, 2024
Aarhus University
CREATES
"""
# ======================================================================================================================

# Importing libraries
import numpy as np


def fDummies(x):
    """
    ARGUMENTS
        * x: Dataframe

    Returns
        * Delta_1: Country dummies
        * Delta_2: Time dummies
    """

    N = x.shape[1]
    T = x.shape[0]

    where_mat = x.isna().values.T

    for t in range(T):
        idx = np.where(~where_mat[:, t])
        idx = np.reshape(idx, (-1,))

        D_t = np.eye(N)

        D_t = D_t[idx, :]

        if t == 0:
            Delta_1 = D_t

            Delta_2 = D_t @ np.ones((N, 1))

        else:
            Delta_1 = np.vstack([Delta_1, D_t])

            Delta_2 = np.hstack((Delta_2, np.zeros((np.shape(Delta_2)[0], 1))))

            Delta_2_tmp = D_t @ np.ones((N, 1))
            Delta_2_tmp = np.hstack((np.zeros((np.shape(Delta_2_tmp)[0], t)), Delta_2_tmp))

            Delta_2 = np.vstack((Delta_2, Delta_2_tmp))

    return Delta_1, Delta_2