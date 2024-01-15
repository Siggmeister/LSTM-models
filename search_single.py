# imports

import sys
import os
import pandas as pd
import numpy as np
# import seaborn as sns
from matplotlib import pyplot as plt
from copy import copy

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from numpy import concatenate
from math import sqrt
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras.backend as K
from sklearn.preprocessing import OneHotEncoder

#f = np.loadtxt('parameters_half.txt', dtype=float)

#n = int(f[int(sys.argv[1]) - 1][0])
#u = int(f[int(sys.argv[1]) - 1][1])
#d = f[int(sys.argv[1]) - 1][2]
#fold = int(f[int(sys.argv[1]) - 1][3])
#lf_num = int(f[int(sys.argv[1]) - 1][4])


def weighted_mse_np(y_true, y_pred):
    """
    Modified Mean Squared Error loss function that weights the square of the error by y_true.

    Parameters:
    y_true (numpy.ndarray): True values.
    y_pred (numpy.ndarray): Predicted values.

    Returns:
    float: Computed weighted mean squared error loss.
    """
    return np.mean(y_true * np.square(y_pred - y_true))


def gaussian_nll(ytrue, ypreds):
    """Keras implmementation of multivariate Gaussian negative loglikelihood loss function.
    This implementation implies diagonal covariance matrix.

    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)

    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples

    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam')

    """

    n_dims = int(int(ypreds.shape[1]) / 2)
    mu = ypreds[:, 0:n_dims]
    logsigma = ypreds[:, n_dims:]

    mse = -0.5 * K.sum(K.square((ytrue - mu) / K.exp(logsigma)), axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5 * n_dims * np.log(2 * np.pi)

    log_likelihood = mse + sigma_trace + log2pi

    return K.mean(-log_likelihood)


def weighted_mse(y_true, y_pred):
    """
    Modified Mean Squared Error loss function that weights the square of the error by y_true.

    Parameters:
    y_true (tensor): True values.
    y_pred (tensor): Predicted values.

    Returns:
    tensor: Computed weighted mean squared error loss.
    """
    return K.mean(y_true * K.square(y_pred - y_true), axis=-1)

def create_multivariate_dataset(dataset, look_back=24, target_index=0):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 1:])
        Y.append(dataset[i + look_back - 1, target_index])
    return np.array(X), np.array(Y)


def split_dataset(dataset, val_start_idx):
    """
    Splits the dataset into train, validation, and test sets.
    dataset: The complete dataset.
    val_start_idx: The starting index for the validation set.
    """
    total_len = len(dataset)
    val_size = int(total_len * 0.1)
    test_size = val_size  # 10% for test set

    # Check if val_start_idx is valid
    if val_start_idx < 0 or (val_start_idx + val_size + test_size) > total_len:
        raise ValueError("Invalid start index for validation set")

    # Define the indices for the validation and test sets
    val_end_idx = val_start_idx + val_size
    test_start_idx = val_end_idx

    # Split the dataset
    validation = dataset[val_start_idx:val_end_idx, :]
    test = dataset[test_start_idx:test_start_idx + test_size, :]

    # The training set is everything outside the validation and test sets
    train = np.concatenate((dataset[:val_start_idx, :], dataset[test_start_idx + test_size:, :]), axis=0)

    return train, validation, test


def create_lstm(n, u, d, lf):
    tf.random.set_seed(7)

    data = pd.read_csv("data_prepared.csv")

    dataset = data[["El Energi", "El Kostnad", "Lufttemperatur", 'Tidligere', 'Antall']].values

    encoder = OneHotEncoder(sparse=False)
    encoded_month = encoder.fit_transform(data[['Måned']])
    encoded_day = encoder.fit_transform(data[['Dag']])
    encoded_weekday = encoder.fit_transform(data[['Ukedag']])
    encoded_hour = encoder.fit_transform(data[['Time']])

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    #dataset = scaler.fit_transform(dataset)
    dataset = concatenate((dataset, encoded_month, encoded_day, encoded_weekday, encoded_hour), axis=1)
    dataset = scaler.fit_transform(dataset)

    # Define the size of each dataset
    train_size = int(len(dataset) * 0.8)
    validation_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - validation_size

    # Split the dataset
    train = dataset[0:train_size, :]
    validation = dataset[train_size:train_size + validation_size, :]
    test = dataset[train_size + validation_size:, :]

    # Specify the index of the target variable in your dataset
    # For example, if your target variable is the first column, target_index = 0
    target_index = 0
    look_back = 24
    train_X, train_y = create_multivariate_dataset(train, look_back, target_index)
    val_X, val_y = create_multivariate_dataset(validation, look_back, target_index)
    test_X, test_y = create_multivariate_dataset(test, look_back, target_index)


    name_of_lf = lf.__name__ if not isinstance(lf, str) else lf

    name = f'best-{name_of_lf}-{n}-{u}-{d}.h5'

    callbacks = [
        EarlyStopping(patience=100, verbose=0),
        ModelCheckpoint(name, verbose=0, save_best_only=True, save_weights_only=True)
    ]

    model = Sequential()

    if n > 1:

        for _ in range(n - 1):
            model.add(LSTM(u, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
            model.add(Dropout(d))

    model.add(LSTM(u, return_sequences=False, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(d))

    if lf is gaussian_nll:
        model.add(Dense(2))
    else:
        model.add(Dense(1))
    
    model.compile(loss=lf, optimizer='adam')

    history = model.fit(train_X, train_y, epochs=1000, callbacks=callbacks, batch_size=64,
                        validation_data=(val_X, val_y), verbose=0,
                        shuffle=False)

    # load the best model
    model.load_weights(name)

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend();
    plt.savefig(f'{name}.png')
    

    yhat = model.predict(test_X)

    if lf is gaussian_nll:
        yhat = yhat[:, 0].reshape(yhat.shape[0], 1)

    # Assuming 'predicted' contains your model's predictions
    # and 'scaler' is your MinMaxScaler instance

    # Create a dummy array with the same shape as your input features
    dummy = np.zeros((len(yhat), 79))

    # Replace the column corresponding to the predicted feature
    # Let's assume your target feature is the first column
    dummy[:, 0] = yhat.reshape(-1)

    # Inverse transform
    rescaled = scaler.inverse_transform(dummy)

    # Extract the predictions, now back in the original scale
    inv_yhat = rescaled[:, 0]

    dummy2 = np.zeros((len(test_y), 79))

    dummy2[:, 0] = test_y.reshape(-1)

    # Inverse transform
    rescaled2 = scaler.inverse_transform(dummy2)

    # Extract the predictions, now back in the original scale
    inv_y = rescaled2[:, 0]

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

    # calculate MSE
    mse = mean_squared_error(inv_y, inv_yhat)

    # calculate MAE
    mae = mean_absolute_error(inv_y, inv_yhat)

    # calculate MAPE
    mape = mean_absolute_percentage_error(inv_y, inv_yhat)

    # calculate wMSE
    wmse = weighted_mse_np(inv_y, inv_yhat)


    return print(rmse, mse, mae, mape, wmse, n, u, d, name_of_lf)



if __name__ == "__main__":
    
    #lf = weighted_mse
    #n, u, d = 1, 64, 0.0

    #lf = gaussian_nll
    #n, u, d = 1, 64, 0.0
    
    #lf = 'mse'
    #n, u, d = 1, 128, 0.5

    lf = 'mape'
    n, u, d = 5, 64, 0.4
    
    create_lstm(n, u, d, lf)
    