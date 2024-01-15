# ======================================================================================================================
"""
This script is used to create a class for the static neural network model for the standard deviation used to create
prediction intervals.

By Sebastian Jensen
Jan, 2024
Aarhus University
CREATES
"""
# ======================================================================================================================

# Importing libraries
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import he_normal, Zeros
from tensorflow.keras.backend import sigmoid, softplus
from tensorflow.keras.layers import Activation
from tensorflow.python.keras.utils.generic_utils import get_custom_objects

from tensorflow.keras.callbacks import EarlyStopping


# %% Creating Swish activation function
def swish(x, beta=1):
    """
    Swish activation function.

    ARGUMENTS
        * x:    input variable.
        * beta: hyperparameter of the Swish activation function.

    Returns
        * Swish activation function applied to x.
    """

    return x * sigmoid(beta * x)


# Registering custom object
get_custom_objects().update({'swish': Activation(swish)})


# %% Creating custom softplus activation function
def sofplus_cust(x, constant=1e-6, factor=0.001):
    """
    Custom softplus activation function.

    ARGUMENTS
        * x:        input variable.
        * constant: constant.
        * factor:   factor.

    Returns
        * Custom softplus activation function applied to x.
    """

    return constant + softplus(factor * x)


# Registering custom object
get_custom_objects().update({'softplus_cust': Activation(sofplus_cust)})


# %% Negative log-likelihood/MSE
def nll_vec():

    def loss(y_true, y_pred):

        return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=1)

    return loss


# %% Creating matrixation layer
class Matrixize(Layer):
    """
    Layer that matrixizes the second dimension of inputs.
    """

    def __init__(self, N, T, noObs, mask, **kwargs):
        super(Matrixize, self).__init__(**kwargs)
        self.N = N
        self.T = T
        self.noObs = noObs
        self.mask = mask

    def call(self, x):
        where = ~self.mask
        indices = tf.cast(tf.where(where), tf.int32)
        scatter = tf.scatter_nd(indices, tf.reshape(x, (-1,)), shape=tf.shape(self.mask))
        scatter = tf.cast(scatter, dtype=np.float64)

        indices = tf.cast(tf.where(~where), tf.int32)
        x_nan = tf.ones(self.N * self.T - self.noObs) * np.nan
        scatter_nan = tf.scatter_nd(indices, x_nan, shape=tf.shape(self.mask))
        scatter_nan = tf.cast(scatter_nan, dtype=np.float64)

        return scatter + scatter_nan

    def compute_output_shape(self, input_shape):
        return [(1, self.T, self.N)]


# %% Creating extend layer
class Extend(Layer):
    """
    Layer that extends the second dimension of inputs.
    """

    def __init__(self, mask, **kwargs):
        super(Extend, self).__init__(**kwargs)
        self.mask = mask

    def call(self, x):
        where = ~self.mask
        indices = tf.cast(tf.where(where), tf.int32)
        scatter = tf.scatter_nd(indices, tf.reshape(x, (-1,)), shape=tf.shape(self.mask))
        scatter = tf.cast(scatter, dtype=np.float64)

        indices = tf.cast(tf.where(~where), tf.int32)
        mask_tmp = tf.cast(self.mask, tf.int32)
        x_nan = tf.ones(tf.reduce_sum(mask_tmp)) * np.nan
        scatter_nan = tf.scatter_nd(indices, x_nan, shape=tf.shape(self.mask))
        scatter_nan = tf.cast(scatter_nan, dtype=np.float64)

        return scatter + scatter_nan

    def compute_output_shape(self, input_shape):
        return [(1, self.T, 1)]


# %% Creating model class
class static_model_sd:
    """
    Class implementing the static neural network model for the standard deviation.
    """

    def __init__(self, nodes, x_train, y_train, pop_train, formulation='global'):
        """
        Instantiating class.

        ARGUMENT
            * nodes:          tuple defining the model architecture.
            * x_train:        dict of TxN_r dataframes of input data (aligned) with a key for each region.
            * y_train:        dict of TxN_r dataframes of target data (aligned) with a key for each region.
            * pop_train:      dict of TxN_r dataframes of population data (aligned) with a key for each region.
            * formulation:    str determining the formulation of the model. Must be one of 'global' or 'regional' or 'national'.

        NB: regions are inferred from the keys of x_train and y_train.
        """

        # Initialization
        self.nodes = nodes
        self.Depth = len(self.nodes)
        self.x_train = x_train
        self.y_train = y_train
        self.pop_train = pop_train
        self.formulation = formulation

        self.individuals = {}
        self.N = {}
        self.noObs = {}

        self.time_periods_na = {}
        self.time_periods_not_na = {}

        self.in_sample_pred = {}

        self.R2 = {}
        self.MSE = {}

        self.Min = {}
        self.Max = {}
        self.quant025 = {}
        self.quant05 = {}
        self.quant95 = {}
        self.quant975 = {}

        self.x_train_np = {}
        self.x_train_df = {}
        self.y_train_df = {}

        self.x_train_transf = {}
        self.y_train_transf = {}
        self.pop_train_transf = {}

        self.x_train_transf_vec = {}
        self.y_train_transf_vec = {}
        self.pop_train_transf_vec = {}

        self.mask = {}
        self.mask_vec = {}

        self.losses = {}

        # Preparing data
        self.regions = list(self.x_train.keys())
        self.no_regions = len(self.regions)

        self.T = self.x_train[self.regions[0]].shape[0]
        self.time_periods = self.x_train[self.regions[0]].index.values

        for region in self.regions:
            self.individuals[region] = x_train[region].columns
            self.time_periods_not_na[region] = np.sum(~np.isnan(self.x_train[region]), axis=1) > 0
            self.time_periods_na[region] = np.sum(~self.time_periods_not_na[region])

            self.N[region] = len(self.individuals[region])

            self.x_train_np[region] = np.array(np.log(self.x_train[region] / self.pop_train[region]))
            self.x_train_df[region] = np.log(self.x_train[region] / self.pop_train[region])
            self.y_train_df[region] = self.y_train[region]

            self.noObs[region] = self.N[region] * self.T - np.isnan(self.x_train_np[region]).sum()

            self.Min[region] = np.nanmin(self.x_train_np[region])
            self.Max[region] = np.nanmax(self.x_train_np[region])
            self.quant025[region] = np.nanquantile(self.x_train_np[region], 0.025)
            self.quant05[region] = np.nanquantile(self.x_train_np[region], 0.05)
            self.quant95[region] = np.nanquantile(self.x_train_np[region], 0.95)
            self.quant975[region] = np.nanquantile(self.x_train_np[region], 0.975)

            self.x_train_transf[region] = self.x_train[region].copy()
            self.y_train_transf[region] = self.y_train[region].copy()
            self.pop_train_transf[region] = self.pop_train[region].copy()

            self.x_train_transf[region] = np.array(np.log(self.x_train_transf[region] / self.pop_train_transf[region]))
            self.y_train_transf[region] = np.array(self.y_train_transf[region])

            self.x_train_transf_vec[region] = np.reshape(self.x_train_transf[region], (-1, 1))
            self.y_train_transf_vec[region] = np.reshape(self.y_train_transf[region], (-1, 1))

            self.mask[region] = np.isnan(self.x_train_transf[region])
            self.mask_vec[region] = np.reshape(~self.mask[region], (-1, 1))

            self.x_train_transf_vec[region] = self.x_train_transf_vec[region][self.mask_vec[region]]
            self.y_train_transf_vec[region] = self.y_train_transf_vec[region][self.mask_vec[region]]

            for individual in self.individuals[region]:
                pos = np.where(self.individuals[region] == individual)[0][0]

                self.Min[individual] = np.nanmin(self.x_train_np[region][:, pos])
                self.Max[individual] = np.nanmax(self.x_train_np[region][:, pos])
                self.quant025[individual] = np.nanquantile(self.x_train_np[region][:, pos], 0.025)
                self.quant05[individual] = np.nanquantile(self.x_train_np[region][:, pos], 0.05)
                self.quant95[individual] = np.nanquantile(self.x_train_np[region][:, pos], 0.95)
                self.quant975[individual] = np.nanquantile(self.x_train_np[region][:, pos], 0.975)

                self.mask_vec[individual] = self.mask[region][:, pos]
                self.x_train_transf_vec[individual] = self.x_train_transf[region][:, pos][~self.mask_vec[individual]]
                self.y_train_transf_vec[individual] = self.y_train_transf[region][:, pos][~self.mask_vec[individual]]
                self.time_periods_not_na[individual] = np.sum(~self.mask_vec[individual])
                self.time_periods_na[individual] = np.sum(self.mask_vec[individual])
                self.noObs[individual] = self.time_periods_not_na[individual]

            if region == self.regions[0]:
                self.individuals['global'] = list(self.individuals[region])
                self.time_periods_not_na['global'] = np.sum(~np.isnan(self.x_train[region]), axis=1) > 0

                self.x_train_transf['global'] = self.x_train[self.regions[0]].copy()
                self.y_train_transf['global'] = self.y_train[self.regions[0]].copy()
                self.pop_train_transf['global'] = self.pop_train[self.regions[0]].copy()

            else:
                self.individuals['global'] = self.individuals['global'] + list(self.individuals[region])
                self.time_periods_not_na['global'] = self.time_periods_not_na['global'] | (np.sum(~np.isnan(self.x_train[region]), axis=1) > 0)

                self.x_train_transf['global'] = pd.concat([self.x_train_transf['global'], self.x_train[region]], axis=1)
                self.y_train_transf['global'] = pd.concat([self.y_train_transf['global'], self.y_train[region]], axis=1)
                self.pop_train_transf['global'] = pd.concat([self.pop_train_transf['global'], self.pop_train[region]], axis=1)

        self.time_periods_na['global'] = np.sum(~self.time_periods_not_na['global'])

        self.N['global'] = np.sum(list(self.N.values()))

        self.x_train_np['global'] = np.array(np.log(self.x_train_transf['global'] / self.pop_train_transf['global']))
        self.noObs['global'] = self.N['global'] * self.T - np.isnan(self.x_train_np['global']).sum()

        self.Min['global'] = np.nanmin(self.x_train_np['global'])
        self.Max['global'] = np.nanmax(self.x_train_np['global'])
        self.quant025['global'] = np.nanquantile(self.x_train_np['global'], 0.025)
        self.quant05['global'] = np.nanquantile(self.x_train_np['global'], 0.05)
        self.quant95['global'] = np.nanquantile(self.x_train_np['global'], 0.95)
        self.quant975['global'] = np.nanquantile(self.x_train_np['global'], 0.975)

        self.x_train_df['global'] = np.log(self.x_train_transf['global'] / self.pop_train_transf['global'])
        self.x_train_transf['global'] = np.array(np.log(self.x_train_transf['global'] / self.pop_train_transf['global']))
        self.y_train_transf['global'] = np.array(self.y_train_transf['global'])

        self.mask['global'] = np.isnan(self.x_train_transf['global'])
        self.mask_vec['global'] = np.reshape(~self.mask['global'], (-1, 1))

        self.x_train_transf_vec['global'] = np.reshape(self.x_train_transf['global'], (-1, 1))
        self.y_train_transf_vec['global'] = np.reshape(self.y_train_transf['global'], (-1, 1))

        self.x_train_transf_vec['global'] = self.x_train_transf_vec['global'][self.mask_vec['global']]
        self.y_train_transf_vec['global'] = self.y_train_transf_vec['global'][self.mask_vec['global']]

        # Setting up the model
        if formulation == 'national':
            # %% Initialization
            input_x_country = [None] * self.N['global']
            input_last = [None] * self.N['global']
            output = [None] * self.N['global']

            self.inputs = [None] * self.N['global']
            self.targets = [None] * self.N['global']

            self.loss_list = [None] * self.N['global']

            self.output_layer = [None] * self.N['global']

            # Building architecture
            hidden_1 = [None] * self.N['global']

            self.hidden_1_layer = Dense(self.nodes[0], activation='swish', use_bias=True, kernel_initializer=he_normal(), bias_initializer=Zeros())

            if self.Depth > 1:
                hidden_2 = [None] * self.N['global']

                self.hidden_2_layer = Dense(self.nodes[1], activation='swish', use_bias=True, kernel_initializer=he_normal(), bias_initializer=Zeros())

                if self.Depth > 2:
                    hidden_3 = [None] * self.N['global']

                    self.hidden_3_layer = Dense(self.nodes[2], activation='swish', use_bias=True, kernel_initializer=he_normal(), bias_initializer=Zeros())

            # Creating the forward pass
            for i in range(self.N['global']):
                input_x_country[i] = Input(batch_input_shape=(1, self.noObs[self.individuals['global'][i]], 1))

                self.inputs[i] = tf.reshape(tf.convert_to_tensor(self.x_train_transf_vec[self.individuals['global'][i]]), (1, self.noObs[self.individuals['global'][i]], 1))
                self.targets[i] = tf.reshape(tf.convert_to_tensor(self.y_train_transf_vec[self.individuals['global'][i]]), (1, self.noObs[self.individuals['global'][i]], 1))

                self.loss_list[i] = nll_vec()

                hidden_1[i] = self.hidden_1_layer(input_x_country[i])

                if self.Depth > 1:
                    hidden_2[i] = self.hidden_2_layer(hidden_1[i])

                    if self.Depth > 2:
                        hidden_3[i] = self.hidden_3_layer(hidden_2[i])

                        input_last[i] = hidden_3[i]

                    else:
                        input_last[i] = hidden_2[i]

                else:
                    input_last[i] = hidden_1[i]

                self.output_layer[i] = Dense(1, activation='softplus_cust', use_bias=True, kernel_initializer=he_normal(), bias_initializer=Zeros())

                output[i] = self.output_layer[i](input_last[i])

            # Compiling the model
            self.model = Model(inputs=input_x_country, outputs=output)

            # Counting number of parameters
            self.m = self.hidden_1_layer.count_params()

            if self.Depth > 1:
                self.m = self.m + self.hidden_2_layer.count_params()

                if self.Depth > 2:
                    self.m = self.m + self.hidden_3_layer.count_params()

            self.m_alt = self.m

            for i in range(self.N['global']):
                self.m = self.m + self.output_layer[i].count_params()

            # Setting up prediction model
            input_x_pred = [None] * self.N['global']
            input_last_pred = [None] * self.N['global']
            output_pred = [None] * self.N['global']

            hidden_1_pred = [None] * self.N['global']

            if self.Depth > 1:
                hidden_2_pred = [None] * self.N['global']

                if self.Depth > 2:
                    hidden_3_pred = [None] * self.N['global']

            self.model_pred = {}

            for i in range(self.N['global']):
                input_x_pred[i] = Input(batch_input_shape=(1, None, 1))

                hidden_1_pred[i] = self.hidden_1_layer(input_x_pred[i])

                if self.Depth > 1:
                    hidden_2_pred[i] = self.hidden_2_layer(hidden_1_pred[i])

                    if self.Depth > 2:
                        hidden_3_pred[i] = self.hidden_3_layer(hidden_2_pred[i])
                        input_last_pred[i] = hidden_3_pred[i]

                    else:
                        input_last_pred[i] = hidden_2_pred[i]

                else:
                    input_last_pred[i] = hidden_1_pred[i]

                output_pred[i] = self.output_layer[i](input_last_pred[i])

            for i in range(self.N['global']):
                self.model_pred[self.individuals['global'][i]] = Model(inputs=input_x_pred[i], outputs=output_pred[i])

        elif formulation == 'regional':
            # %% Initialization
            inputs_x = [None] * self.no_regions
            input_last = [None] * self.no_regions
            output = [None] * self.no_regions

            self.inputs = [None] * self.no_regions
            self.inputs_X = [None] * self.no_regions
            self.targets = [None] * self.no_regions

            self.loss_list = [None] * self.no_regions

            self.output_layer = [None] * self.no_regions

            # Building architecture
            hidden_1 = [None] * self.no_regions

            self.hidden_1_layer = Dense(self.nodes[0], activation='swish', use_bias=True, kernel_initializer=he_normal(), bias_initializer=Zeros())

            if self.Depth > 1:
                hidden_2 = [None] * self.no_regions

                self.hidden_2_layer = Dense(self.nodes[1], activation='swish', use_bias=True, kernel_initializer=he_normal(), bias_initializer=Zeros())

                if self.Depth > 2:
                    hidden_3 = [None] * self.no_regions

                    self.hidden_3_layer = Dense(self.nodes[2], activation='swish', use_bias=True, kernel_initializer=he_normal(), bias_initializer=Zeros())

            # Creating the forward pass
            for i in range(self.no_regions):
                inputs_x[i] = Input(batch_input_shape=(1, self.noObs[self.regions[i]], 1))

                self.inputs_X[i] = tf.reshape(tf.convert_to_tensor(self.x_train_transf_vec[self.regions[i]]), (1, self.noObs[self.regions[i]], 1))
                self.targets[i] = tf.reshape(tf.convert_to_tensor(self.y_train_transf_vec[self.regions[i]]), (1, self.noObs[self.regions[i]], 1))

                self.loss_list[i] = nll_vec()

                hidden_1[i] = self.hidden_1_layer(inputs_x[i])

                if self.Depth > 1:
                    hidden_2[i] = self.hidden_2_layer(hidden_1[i])

                    if self.Depth > 2:
                        hidden_3[i] = self.hidden_3_layer(hidden_2[i])

                        input_last[i] = hidden_3[i]

                    else:
                        input_last[i] = hidden_2[i]

                else:
                    input_last[i] = hidden_1[i]

                self.output_layer[i] = Dense(1, activation='softplus_cust', use_bias=True, kernel_initializer=he_normal(), bias_initializer=Zeros())

                output[i] = self.output_layer[i](input_last[i])

            # Compiling the model
            self.model = Model(inputs=inputs_x, outputs=output)

            self.inputs = self.inputs_X

            # Counting number of parameters
            self.m = self.hidden_1_layer.count_params()

            if self.Depth > 1:
                self.m = self.m + self.hidden_2_layer.count_params()

                if self.Depth > 2:
                    self.m = self.m + self.hidden_3_layer.count_params()

            self.m_alt = self.m

            for i in range(self.no_regions):
                self.m = self.m + self.output_layer[i].count_params()

            # Setting up prediction model
            input_x_pred = [None] * self.no_regions
            output_pred = [None] * self.no_regions
            input_last_pred = [None] * self.no_regions

            hidden_1_pred = [None] * self.no_regions

            if self.Depth > 1:
                hidden_2_pred = [None] * self.no_regions

                if self.Depth > 2:
                    hidden_3_pred = [None] * self.no_regions

            self.model_pred = {}

            for i in range(self.no_regions):
                input_x_pred[i] = Input(batch_input_shape=(1, None, 1))

                hidden_1_pred[i] = self.hidden_1_layer(input_x_pred[i])

                if self.Depth > 1:
                    hidden_2_pred[i] = self.hidden_2_layer(hidden_1_pred[i])

                    if self.Depth > 2:
                        hidden_3_pred[i] = self.hidden_3_layer(hidden_2_pred[i])
                        input_last_pred[i] = hidden_3_pred[i]

                    else:
                        input_last_pred[i] = hidden_2_pred[i]

                else:
                    input_last_pred[i] = hidden_1_pred[i]

                output_pred[i] = self.output_layer[i](input_last_pred[i])

            for i in range(self.no_regions):
                self.model_pred[self.regions[i]] = Model(inputs=input_x_pred[i], outputs=output_pred[i])

        else:
            # %% Initialization
            input_x = Input(batch_input_shape=(1, self.noObs['global'], 1))

            self.input_X = tf.reshape(tf.convert_to_tensor(self.x_train_transf_vec['global']), (1, self.noObs['global'], 1))
            self.targets = tf.reshape(tf.convert_to_tensor(self.y_train_transf_vec['global']), (1, self.noObs['global'], 1))

            # Creating the forward pass
            self.hidden_1_layer = Dense(self.nodes[0], activation='swish', use_bias=True, kernel_initializer=he_normal(), bias_initializer=Zeros())

            hidden_1 = self.hidden_1_layer(input_x)

            if self.Depth > 1:
                self.hidden_2_layer = Dense(self.nodes[1], activation='swish', use_bias=True, kernel_initializer=he_normal(), bias_initializer=Zeros())

                hidden_2 = self.hidden_2_layer(hidden_1)

                if self.Depth > 2:
                    self.hidden_3_layer = Dense(self.nodes[2], activation='swish', use_bias=True, kernel_initializer=he_normal(), bias_initializer=Zeros())

                    hidden_3 = self.hidden_3_layer(hidden_2)

                    input_last = hidden_3

                else:
                    input_last = hidden_2

            else:
                input_last = hidden_1

            self.output_layer = Dense(1, activation='softplus_cust', use_bias=True, kernel_initializer=he_normal(), bias_initializer=Zeros())

            output = self.output_layer(input_last)

            # Compiling the model
            self.model = Model(inputs=input_x, outputs=output)

            self.inputs = self.input_X

            # Counting number of parameters
            self.m = self.hidden_1_layer.count_params()

            if self.Depth > 1:
                self.m = self.m + self.hidden_2_layer.count_params()

                if self.Depth > 2:
                    self.m = self.m + self.hidden_3_layer.count_params()

            self.m_alt = self.m

            self.m = self.m + self.output_layer.count_params()

            # Setting up prediction model
            input_x_pred = Input(batch_input_shape=(1, None, 1))

            hidden_1_pred = self.hidden_1_layer(input_x_pred)

            if self.Depth > 1:
                hidden_2_pred = self.hidden_2_layer(hidden_1_pred)

                if self.Depth > 2:
                    hidden_3_pred = self.hidden_3_layer(hidden_2_pred)
                    input_last_pred = hidden_3_pred

                else:
                    input_last_pred = hidden_2_pred

            else:
                input_last_pred = hidden_1_pred

            output_pred = self.output_layer(input_last_pred)

            self.model_pred = Model(inputs=input_x_pred, outputs=output_pred)

    def fit(self, lr=0.001, min_delta=1e-6, patience=100, verbose=1):
        """
        Fitting the model.

        ARGUMENTS
            * lr:            initial learning rate of the Adam optimizer.
            * min_delta:     tolerance to be used for optimization.
            * patience:      patience to be used for optimization.
            * verbose:       verbosity mode for optimization.
        """

        if self.formulation == 'national':
            self.model.compile(optimizer=Adam(lr), loss=self.loss_list, loss_weights=[1 / self.N['global']] * self.N['global'])

        elif self.formulation == 'regional':
            self.model.compile(optimizer=Adam(lr), loss=self.loss_list, loss_weights=[1 / self.no_regions] * self.no_regions)

        else:
            self.model.compile(optimizer=Adam(lr), loss=nll_vec())

        callbacks = [EarlyStopping(monitor='loss', mode='min', min_delta=min_delta, patience=patience, restore_best_weights=True, verbose=verbose)]

        self.model.fit(self.inputs, self.targets, callbacks=callbacks, batch_size=1, epochs=int(1e6), verbose=verbose, shuffle=False)

        self.losses = self.model.history.history
        self.epochs = self.model.history.epoch

        self.params = self.model.get_weights()

    def load_params(self, filepath):
        """
        Loading model parameters.

         ARGUMENTS
            * filepath: string containing path/name of saved file.
        """

        self.model.load_weights(filepath)
        self.params = self.model.get_weights()

    def save_params(self, filepath):
        """
        Saving model parameters.

         ARGUMENTS
            * filepath: string containing path/name of file to be saved.
        """

        self.model.save_weights(filepath)

    def in_sample_predictions(self):
        """
        Making in-sample predictions.

        """
        in_sample_preds = self.model(self.inputs)
        sigma2_tmp = 0
        noObs_tmp = 0
        N_agg = 0
        country_counter = 0

        if self.formulation == 'global':
            Mask = tf.reshape(tf.convert_to_tensor(self.mask['global']), (1, self.T, self.N['global']))
            in_sample_preds_mat = Matrixize(N=self.N['global'], T=self.T, noObs=self.noObs['global'], mask=Mask)(in_sample_preds)

        for region in self.regions:
            self.in_sample_pred[region] = self.y_train[region].copy()

            if self.formulation == 'national':
                for i in range(self.N[region]):
                    Mask = tf.reshape(tf.convert_to_tensor(self.mask[region][:, i]), (1, self.T, 1))
                    in_sample_preds_ext = Extend(mask=Mask)(in_sample_preds[country_counter][0, :, 0])

                    self.in_sample_pred[region].iloc[:, i] = np.reshape(np.array(in_sample_preds_ext), (-1,))

                    country_counter = country_counter + 1

            elif self.formulation == 'regional':
                Mask = tf.reshape(tf.convert_to_tensor(self.mask[region]), (1, self.T, self.N[region]))
                in_sample_preds_mat = Matrixize(N=self.N[region], T=self.T, noObs=self.noObs[region], mask=Mask)(in_sample_preds[self.regions.index(region)])

                self.in_sample_pred[region].iloc[:, :] = np.array(in_sample_preds_mat[0, :, :])

            else:
                self.in_sample_pred[region].iloc[:, :] = np.array(in_sample_preds_mat[0, :, N_agg:N_agg+self.N[region]])
                N_agg = N_agg + self.N[region]

            if self.regions.index(region) == 0:
                in_sample_pred_global = self.in_sample_pred[region]
                in_sample_global = self.y_train_df[region]
            else:
                in_sample_pred_global = pd.concat([in_sample_pred_global, self.in_sample_pred[region]], axis=1)
                in_sample_global = pd.concat([in_sample_global, self.y_train_df[region]], axis=1)

            mean_tmp = np.nanmean(np.reshape(np.array(self.y_train_df[region]), (-1)))

            SSR = np.sum(np.sum((self.y_train_df[region] - self.in_sample_pred[region]) ** 2))
            SST = np.sum(np.sum((self.y_train_df[region] - mean_tmp) ** 2))
            self.R2[region] = 1 - SSR / SST
            self.MSE[region] = SSR / self.noObs[region]

            if self.formulation == 'national':
                for i in range(self.N[region]):
                    SSR = np.sum((self.y_train_df[region].iloc[:, i] - self.in_sample_pred[region].iloc[:, i]) ** 2)
                    noObs_tmp = np.sum(~np.isnan(self.y_train_df[region]).iloc[:, i])

                    sigma2_tmp = sigma2_tmp + (1 / self.N['global']) * (SSR / noObs_tmp)

            elif self.formulation == 'regional':
                sigma2_tmp = sigma2_tmp + (1 / self.no_regions) * (SSR / self.noObs[region])
                noObs_tmp = noObs_tmp + self.noObs[region]

            else:
                sigma2_tmp = sigma2_tmp + SSR

        mean_tmp = np.nanmean(np.reshape(np.array(in_sample_global), (-1)))

        SSR = np.sum(np.sum((in_sample_global - in_sample_pred_global)**2))
        SST = np.sum(np.sum((in_sample_global - mean_tmp)**2))
        self.R2['global'] = 1 - SSR / SST
        self.MSE['global'] = SSR / self.noObs['global']

        if self.formulation == 'national':
            self.BIC = np.log(sigma2_tmp) + self.m * np.log(self.noObs['global']) / self.noObs['global']

        elif self.formulation == 'regional':
            self.BIC = np.log(sigma2_tmp) + self.m * np.log(noObs_tmp) / noObs_tmp

        else:
            self.BIC = np.log(sigma2_tmp) - np.log(self.noObs['global']) + self.m * np.log(self.noObs['global']) / self.noObs['global']

    def predict(self, x_test, idx=None):
        """
        Making predictions.

        ARGUMENTS
            * x_test:   (-1,1) array of input data.
            * idx:      Name identifying the country/region to be used for making predictions (if national or regional formulation).

        RETURNS
            * pred_df: Dataframe containing predictions.
        """

        x_test_tf = tf.convert_to_tensor(np.reshape(x_test, (1, -1, 1)))

        if self.formulation in ['national', 'regional']:
            pred_np = np.reshape(self.model_pred[idx](x_test_tf), (-1, 1), order='F')

        else:
            pred_np = np.reshape(self.model_pred(x_test_tf), (-1, 1), order='F')

        pred_df = pd.DataFrame(pred_np)
        pred_df.set_index(np.reshape(x_test, (-1,)), inplace=True)

        return pred_df
