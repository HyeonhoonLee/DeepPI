# -*- coding: utf-8 -*-

# Coded by Hyeonhoon Lee, KMD, Ph.D

"""#1. Load libraries"""

import pandas as pd
import numpy as np

import os
import re

import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from datetime import date

sns.set_style('ticks')

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.io as pio

"""#2. Load file"""

SEED = 42

# DATADIR and FILENAME should be included
pre_df = pd.read_csv(os.path.join(DATADIR, FILENAME))
poorsleep_df = pre_df[pre_df['PSQI_global'] > 5]
cleanchart_df = poorsleep_df.dropna(axis=0)

"""#3. Prepare Dataset"""

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(cleanchart_df, test_size=0.2, random_state=SEED)

train_df

test_df

catercol = []
for col in train_df:
  if max(cleanchart_df[col].values) == 1 and min(cleanchart_df[col].values) == 0:
    catercol.append(col)

numercol = []
for col in train_df:
  if col not in catercol:
    numercol.append(col)

train_numer = train_df[numercol]
train_cater = train_df[catercol]

test_numer = test_df[numercol]
test_cater = test_df[catercol]

scaler = StandardScaler()

train_scaled_features = scaler.fit_transform(train_numer.values)
test_scaled_features = scaler.transform(test_numer.values)

train_numer_scaled = pd.DataFrame(train_scaled_features, index=train_numer.index, columns=train_numer.columns)
test_numer_scaled = pd.DataFrame(test_scaled_features, index=test_numer.index, columns=test_numer.columns)

numercol_new=[]
for col in numercol:
  newname = col + '_Scaled'
  numercol_new.append(newname)

train_numer_scaled.columns = numercol_new
test_numer_scaled.columns = numercol_new

train_scaled_df = pd.concat([train_numer_scaled, train_cater], axis=1)
test_scaled_df = pd.concat([test_numer_scaled, test_cater], axis=1)

"""#4. Dimentionality reduction"""

##4.1. PCA
pca_df = train_scaled_df.copy()

pca=PCA(random_state=SEED)
pca.fit(pca_df)

plt.figure(figsize=(12,8))
plt.bar(x=list(range(1, pca_df.shape[1]+1)), height=pca.explained_variance_ratio_,color='green')
plt.xlabel('Components',fontsize=12)
plt.ylim(0,0.08)
plt.xlim(0,100)
plt.ylabel('Variance%',fontsize=12)
plt.show()

exp_var_pca = pca.explained_variance_ratio_

cum_sum_eigenvalues = np.cumsum(exp_var_pca)

plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

pca=PCA(n_components=4, random_state=SEED)
pca.fit(pca_df)

pca_array = pca.transform(pca_df)

##4.2. Autoencoder

X_traindata = np.array(train_scaled_df)

X_traindata.shape

from keras.layers import Dense, Dropout
from keras.models import Sequential, Model
from keras import metrics, Input

from sklearn.model_selection import KFold

import tensorflow as tf
import random as python_random

np.random.seed(SEED)
python_random.seed(SEED)
tf.random.set_seed(SEED)

METRICS = [
    metrics.RootMeanSquaredError(name='rms'),]
BATCH_SIZE = 64
EPOCHS = 100

kf = KFold(n_splits = 10)

input_output_dim = X_traindata.shape[-1]

ENCODING_DIM = 1 #Desired Dimension

def create_model(input_output_dim=160):
    input_ = Input(shape=(input_output_dim,))
    encoded = Dense(units=ENCODING_DIM*8, activation="relu")(input_)
    bottleneck = Dense(units=ENCODING_DIM, 
                       activation="relu")(encoded)
    decoded = Dense(units=ENCODING_DIM*8, 
                    activation="relu")(bottleneck)
    output = Dense(units=input_output_dim, 
                    activation="linear")(decoded)
    autoencoder = Model(inputs=input_, outputs=output)
    return autoencoder

valid_rmse = []
folds = []
encoding_dim = []
epochs = []
last_train_rmse = []
last_val_rmse = []
idx = 0

for train_idx, valid_idx in kf.split(X_traindata):
  X_train, X_valid = X_traindata[train_idx], X_traindata[valid_idx]
  autoencoder = create_model()

  autoencoder.compile(optimizer='adam', loss='mse',
                        metrics=METRICS)
  history = autoencoder.fit(X_train, X_train,
                  batch_size=BATCH_SIZE,
                  validation_data=(X_valid, X_valid),
                  validation_steps=len(X_valid)//BATCH_SIZE,
                  epochs=EPOCHS)
  print(f'fold{idx}')
  autoencoder.save(os.path.join(DATADIR, 'weights', f'hidden{ENCODING_DIM}', 'fold'+str(idx)+'_lastEpoch.h5'))
  train_rmse = history.history['rms'][-1]
  val_rmse = history.history['val_rms'][-1]

  valid_rmse.append(history.history['val_rms'])
  folds.append([idx] * EPOCHS)
  encoding_dim.append([ENCODING_DIM] * EPOCHS)
  epochs.append([i for i in range(EPOCHS)])

  last_train_rmse.append(train_rmse)
  last_val_rmse.append(val_rmse)

  idx+=1
  plt.plot(history.history['val_rms'], label='val'+str(idx))
  plt.title(f'J = {ENCODING_DIM}')
  plt.ylabel('RMSE')
  plt.xlabel('Epochs')

valid_rmse = np.concatenate(valid_rmse)
encoding_dim = np.concatenate(encoding_dim)
folds = np.concatenate(folds)
epochs = np.concatenate(epochs)

data_tuples = list(zip(encoding_dim, folds, epochs, valid_rmse))
rmse_df1 = pd.DataFrame(data_tuples, columns=['Encoding_dim','Folds', 'Epochs', 'Valid_rmse'])
rmse_df1

ENCODING_DIM = 2 #Desired Dimension

def create_model(input_output_dim=160):
    input_ = Input(shape=(input_output_dim,))
    encoded = Dense(units=ENCODING_DIM*8, activation="relu")(input_)
    bottleneck = Dense(units=ENCODING_DIM, 
                       activation="relu")(encoded)
    decoded = Dense(units=ENCODING_DIM*8, 
                    activation="relu")(bottleneck)
    output = Dense(units=input_output_dim, 
                    activation="linear")(decoded)
    autoencoder = Model(inputs=input_, outputs=output)
    return autoencoder

valid_rmse = []
folds = []
encoding_dim = []
epochs = []
last_train_rmse = []
last_val_rmse = []
idx = 0

for train_idx, valid_idx in kf.split(X_traindata):
  X_train, X_valid = X_traindata[train_idx], X_traindata[valid_idx]
  autoencoder = create_model()

  autoencoder.compile(optimizer='adam', loss='mse',
                        metrics=METRICS)
  history = autoencoder.fit(X_train, X_train,
                  batch_size=BATCH_SIZE,
                  validation_data=(X_valid, X_valid),
                  validation_steps=len(X_valid)//BATCH_SIZE,
                  epochs=EPOCHS)
  print(f'fold{idx}')
  autoencoder.save(os.path.join(DATADIR, 'weights', f'hidden{ENCODING_DIM}', 'fold'+str(idx)+'_lastEpoch.h5'))
  train_rmse = history.history['rms'][-1]
  val_rmse = history.history['val_rms'][-1]

  valid_rmse.append(history.history['val_rms'])
  folds.append([idx] * EPOCHS)
  encoding_dim.append([ENCODING_DIM] * EPOCHS)
  epochs.append([i for i in range(EPOCHS)])

  last_train_rmse.append(train_rmse)
  last_val_rmse.append(val_rmse)

  idx+=1
  plt.plot(history.history['val_rms'], label='val'+str(idx))
  plt.title(f'J = {ENCODING_DIM}')
  plt.ylabel('RMSE')
  plt.xlabel('Epochs')

valid_rmse = np.concatenate(valid_rmse)
encoding_dim = np.concatenate(encoding_dim)
folds = np.concatenate(folds)
epochs = np.concatenate(epochs)

data_tuples = list(zip(encoding_dim, folds, epochs, valid_rmse))
rmse_df2 = pd.DataFrame(data_tuples, columns=['Encoding_dim','Folds', 'Epochs', 'Valid_rmse'])
rmse_df2

ENCODING_DIM = 3 #Desired Dimension

def create_model(input_output_dim=160):
    input_ = Input(shape=(input_output_dim,))
    encoded = Dense(units=ENCODING_DIM*8, activation="relu")(input_)
    bottleneck = Dense(units=ENCODING_DIM, 
                       activation="relu")(encoded)
    decoded = Dense(units=ENCODING_DIM*8, 
                    activation="relu")(bottleneck)
    output = Dense(units=input_output_dim, 
                    activation="linear")(decoded)
    autoencoder = Model(inputs=input_, outputs=output)
    return autoencoder

valid_rmse = []
folds = []
encoding_dim = []
epochs = []
last_train_rmse = []
last_val_rmse = []
idx = 0

for train_idx, valid_idx in kf.split(X_traindata):
  X_train, X_valid = X_traindata[train_idx], X_traindata[valid_idx]
  autoencoder = create_model()

  autoencoder.compile(optimizer='adam', loss='mse',
                        metrics=METRICS)
  history = autoencoder.fit(X_train, X_train,
                  batch_size=BATCH_SIZE,
                  validation_data=(X_valid, X_valid),
                  validation_steps=len(X_valid)//BATCH_SIZE,
                  epochs=EPOCHS)
  print(f'fold{idx}')
  autoencoder.save(os.path.join(DATADIR, 'weights', f'hidden{ENCODING_DIM}', 'fold'+str(idx)+'_lastEpoch.h5'))
  train_rmse = history.history['rms'][-1]
  val_rmse = history.history['val_rms'][-1]

  valid_rmse.append(history.history['val_rms'])
  folds.append([idx] * EPOCHS)
  encoding_dim.append([ENCODING_DIM] * EPOCHS)
  epochs.append([i for i in range(EPOCHS)])

  last_train_rmse.append(train_rmse)
  last_val_rmse.append(val_rmse)

  idx+=1
  plt.plot(history.history['val_rms'], label='val'+str(idx))
  plt.title(f'J = {ENCODING_DIM}')
  plt.ylabel('RMSE')
  plt.xlabel('Epochs')

valid_rmse = np.concatenate(valid_rmse)
encoding_dim = np.concatenate(encoding_dim)
folds = np.concatenate(folds)
epochs = np.concatenate(epochs)

data_tuples = list(zip(encoding_dim, folds, epochs, valid_rmse))
rmse_df3 = pd.DataFrame(data_tuples, columns=['Encoding_dim','Folds', 'Epochs', 'Valid_rmse'])
rmse_df3

ENCODING_DIM = 4 #Desired Dimension

def create_model(input_output_dim=160):
    input_ = Input(shape=(input_output_dim,))
    encoded = Dense(units=ENCODING_DIM*8, activation="relu")(input_)
    bottleneck = Dense(units=ENCODING_DIM, 
                       activation="relu")(encoded)
    decoded = Dense(units=ENCODING_DIM*8, 
                    activation="relu")(bottleneck)
    output = Dense(units=input_output_dim, 
                    activation="linear")(decoded)
    autoencoder = Model(inputs=input_, outputs=output)
    return autoencoder

valid_rmse = []
folds = []
encoding_dim = []
epochs = []
last_train_rmse = []
last_val_rmse = []
idx = 0

for train_idx, valid_idx in kf.split(X_traindata):
  X_train, X_valid = X_traindata[train_idx], X_traindata[valid_idx]
  autoencoder = create_model()

  autoencoder.compile(optimizer='adam', loss='mse',
                        metrics=METRICS)
  history = autoencoder.fit(X_train, X_train,
                  batch_size=BATCH_SIZE,
                  validation_data=(X_valid, X_valid),
                  validation_steps=len(X_valid)//BATCH_SIZE,
                  epochs=EPOCHS)
  print(f'fold{idx}')
  autoencoder.save(os.path.join(DATADIR, 'weights', f'hidden{ENCODING_DIM}', 'fold'+str(idx)+'_lastEpoch.h5'))
  train_rmse = history.history['rms'][-1]
  val_rmse = history.history['val_rms'][-1]

  valid_rmse.append(history.history['val_rms'])
  folds.append([idx] * EPOCHS)
  encoding_dim.append([ENCODING_DIM] * EPOCHS)
  epochs.append([i for i in range(EPOCHS)])

  last_train_rmse.append(train_rmse)
  last_val_rmse.append(val_rmse)

  idx+=1
  plt.plot(history.history['val_rms'], label='val'+str(idx))
  plt.title(f'J = {ENCODING_DIM}')
  plt.ylabel('RMSE')
  plt.xlabel('Epochs')

valid_rmse = np.concatenate(valid_rmse)
encoding_dim = np.concatenate(encoding_dim)
folds = np.concatenate(folds)
epochs = np.concatenate(epochs)

data_tuples = list(zip(encoding_dim, folds, epochs, valid_rmse))
rmse_df4 = pd.DataFrame(data_tuples, columns=['Encoding_dim','Folds', 'Epochs', 'Valid_rmse'])
rmse_df4

ENCODING_DIM = 5 #Desired Dimension

def create_model(input_output_dim=160):
    input_ = Input(shape=(input_output_dim,))
    encoded = Dense(units=ENCODING_DIM*8, activation="relu")(input_)
    bottleneck = Dense(units=ENCODING_DIM, 
                       activation="relu")(encoded)
    decoded = Dense(units=ENCODING_DIM*8, 
                    activation="relu")(bottleneck)
    output = Dense(units=input_output_dim, 
                    activation="linear")(decoded)
    autoencoder = Model(inputs=input_, outputs=output)
    return autoencoder

valid_rmse = []
folds = []
encoding_dim = []
epochs = []
last_train_rmse = []
last_val_rmse = []
idx = 0

for train_idx, valid_idx in kf.split(X_traindata):
  X_train, X_valid = X_traindata[train_idx], X_traindata[valid_idx]
  autoencoder = create_model()

  autoencoder.compile(optimizer='adam', loss='mse',
                        metrics=METRICS)
  history = autoencoder.fit(X_train, X_train,
                  batch_size=BATCH_SIZE,
                  validation_data=(X_valid, X_valid),
                  validation_steps=len(X_valid)//BATCH_SIZE,
                  epochs=EPOCHS)
  print(f'fold{idx}')
  autoencoder.save(os.path.join(DATADIR, 'weights', f'hidden{ENCODING_DIM}', 'fold'+str(idx)+'_lastEpoch.h5'))
  train_rmse = history.history['rms'][-1]
  val_rmse = history.history['val_rms'][-1]

  valid_rmse.append(history.history['val_rms'])
  folds.append([idx] * EPOCHS)
  encoding_dim.append([ENCODING_DIM] * EPOCHS)
  epochs.append([i for i in range(EPOCHS)])

  last_train_rmse.append(train_rmse)
  last_val_rmse.append(val_rmse)

  idx+=1
  plt.plot(history.history['val_rms'], label='val'+str(idx))
  plt.title(f'J = {ENCODING_DIM}')
  plt.ylabel('RMSE')
  plt.xlabel('Epochs')

valid_rmse = np.concatenate(valid_rmse)
encoding_dim = np.concatenate(encoding_dim)
folds = np.concatenate(folds)
epochs = np.concatenate(epochs)

data_tuples = list(zip(encoding_dim, folds, epochs, valid_rmse))
rmse_df5 = pd.DataFrame(data_tuples, columns=['Encoding_dim','Folds', 'Epochs', 'Valid_rmse'])
rmse_df5

ENCODING_DIM = 6 #Desired Dimension

def create_model(input_output_dim=160):
    input_ = Input(shape=(input_output_dim,))
    encoded = Dense(units=ENCODING_DIM*8, activation="relu")(input_)
    bottleneck = Dense(units=ENCODING_DIM, 
                       activation="relu")(encoded)
    decoded = Dense(units=ENCODING_DIM*8, 
                    activation="relu")(bottleneck)
    output = Dense(units=input_output_dim, 
                    activation="linear")(decoded)
    autoencoder = Model(inputs=input_, outputs=output)
    return autoencoder

valid_rmse = []
folds = []
encoding_dim = []
epochs = []
last_train_rmse = []
last_val_rmse = []
idx = 0

for train_idx, valid_idx in kf.split(X_traindata):
  X_train, X_valid = X_traindata[train_idx], X_traindata[valid_idx]
  autoencoder = create_model()

  autoencoder.compile(optimizer='adam', loss='mse',
                        metrics=METRICS)
  history = autoencoder.fit(X_train, X_train,
                  batch_size=BATCH_SIZE,
                  validation_data=(X_valid, X_valid),
                  validation_steps=len(X_valid)//BATCH_SIZE,
                  epochs=EPOCHS)
  print(f'fold{idx}')
  autoencoder.save(os.path.join(DATADIR, 'weights', f'hidden{ENCODING_DIM}', 'fold'+str(idx)+'_lastEpoch.h5'))
  train_rmse = history.history['rms'][-1]
  val_rmse = history.history['val_rms'][-1]

  valid_rmse.append(history.history['val_rms'])
  folds.append([idx] * EPOCHS)
  encoding_dim.append([ENCODING_DIM] * EPOCHS)
  epochs.append([i for i in range(EPOCHS)])

  last_train_rmse.append(train_rmse)
  last_val_rmse.append(val_rmse)

  idx+=1
  plt.plot(history.history['val_rms'], label='val'+str(idx))
  plt.title(f'J = {ENCODING_DIM}')
  plt.ylabel('RMSE')
  plt.xlabel('Epochs')

valid_rmse = np.concatenate(valid_rmse)
encoding_dim = np.concatenate(encoding_dim)
folds = np.concatenate(folds)
epochs = np.concatenate(epochs)

data_tuples = list(zip(encoding_dim, folds, epochs, valid_rmse))
rmse_df6 = pd.DataFrame(data_tuples, columns=['Encoding_dim','Folds', 'Epochs', 'Valid_rmse'])
rmse_df6

ENCODING_DIM = 7 #Desired Dimension

def create_model(input_output_dim=160):
    input_ = Input(shape=(input_output_dim,))
    encoded = Dense(units=ENCODING_DIM*8, activation="relu")(input_)
    bottleneck = Dense(units=ENCODING_DIM, 
                       activation="relu")(encoded)
    decoded = Dense(units=ENCODING_DIM*8, 
                    activation="relu")(bottleneck)
    output = Dense(units=input_output_dim, 
                    activation="linear")(decoded)
    autoencoder = Model(inputs=input_, outputs=output)
    return autoencoder

valid_rmse = []
folds = []
encoding_dim = []
epochs = []
last_train_rmse = []
last_val_rmse = []
idx = 0

for train_idx, valid_idx in kf.split(X_traindata):
  X_train, X_valid = X_traindata[train_idx], X_traindata[valid_idx]
  autoencoder = create_model()

  autoencoder.compile(optimizer='adam', loss='mse',
                        metrics=METRICS)
  history = autoencoder.fit(X_train, X_train,
                  batch_size=BATCH_SIZE,
                  validation_data=(X_valid, X_valid),
                  validation_steps=len(X_valid)//BATCH_SIZE,
                  epochs=EPOCHS)
  print(f'fold{idx}')
  autoencoder.save(os.path.join(DATADIR, 'weights', f'hidden{ENCODING_DIM}', 'fold'+str(idx)+'_lastEpoch.h5'))
  train_rmse = history.history['rms'][-1]
  val_rmse = history.history['val_rms'][-1]

  valid_rmse.append(history.history['val_rms'])
  folds.append([idx] * EPOCHS)
  encoding_dim.append([ENCODING_DIM] * EPOCHS)
  epochs.append([i for i in range(EPOCHS)])

  last_train_rmse.append(train_rmse)
  last_val_rmse.append(val_rmse)

  idx+=1
  plt.plot(history.history['val_rms'], label='val'+str(idx))
  plt.title(f'J = {ENCODING_DIM}')
  plt.ylabel('RMSE')
  plt.xlabel('Epochs')

valid_rmse = np.concatenate(valid_rmse)
encoding_dim = np.concatenate(encoding_dim)
folds = np.concatenate(folds)
epochs = np.concatenate(epochs)

data_tuples = list(zip(encoding_dim, folds, epochs, valid_rmse))
rmse_df7 = pd.DataFrame(data_tuples, columns=['Encoding_dim','Folds', 'Epochs', 'Valid_rmse'])
rmse_df7

ENCODING_DIM = 8 #Desired Dimension

def create_model(input_output_dim=160):
    input_ = Input(shape=(input_output_dim,))
    encoded = Dense(units=ENCODING_DIM*8, activation="relu")(input_)
    bottleneck = Dense(units=ENCODING_DIM, 
                       activation="relu")(encoded)
    decoded = Dense(units=ENCODING_DIM*8, 
                    activation="relu")(bottleneck)
    output = Dense(units=input_output_dim, 
                    activation="linear")(decoded)
    autoencoder = Model(inputs=input_, outputs=output)
    return autoencoder

valid_rmse = []
folds = []
encoding_dim = []
epochs = []
last_train_rmse = []
last_val_rmse = []
idx = 0

for train_idx, valid_idx in kf.split(X_traindata):
  X_train, X_valid = X_traindata[train_idx], X_traindata[valid_idx]
  autoencoder = create_model()

  autoencoder.compile(optimizer='adam', loss='mse',
                        metrics=METRICS)
  history = autoencoder.fit(X_train, X_train,
                  batch_size=BATCH_SIZE,
                  validation_data=(X_valid, X_valid),
                  validation_steps=len(X_valid)//BATCH_SIZE,
                  epochs=EPOCHS)
  print(f'fold{idx}')
  autoencoder.save(os.path.join(DATADIR, 'weights', f'hidden{ENCODING_DIM}', 'fold'+str(idx)+'_lastEpoch.h5'))
  train_rmse = history.history['rms'][-1]
  val_rmse = history.history['val_rms'][-1]

  valid_rmse.append(history.history['val_rms'])
  folds.append([idx] * EPOCHS)
  encoding_dim.append([ENCODING_DIM] * EPOCHS)
  epochs.append([i for i in range(EPOCHS)])

  last_train_rmse.append(train_rmse)
  last_val_rmse.append(val_rmse)

  idx+=1
  plt.plot(history.history['val_rms'], label='val'+str(idx))
  plt.title(f'J = {ENCODING_DIM}')
  plt.ylabel('RMSE')
  plt.xlabel('Epochs')

valid_rmse = np.concatenate(valid_rmse)
encoding_dim = np.concatenate(encoding_dim)
folds = np.concatenate(folds)
epochs = np.concatenate(epochs)

data_tuples = list(zip(encoding_dim, folds, epochs, valid_rmse))
rmse_df8 = pd.DataFrame(data_tuples, columns=['Encoding_dim','Folds', 'Epochs', 'Valid_rmse'])
rmse_df8

ENCODING_DIM = 9 #Desired Dimension

def create_model(input_output_dim=160):
    input_ = Input(shape=(input_output_dim,))
    encoded = Dense(units=ENCODING_DIM*8, activation="relu")(input_)
    bottleneck = Dense(units=ENCODING_DIM, 
                       activation="relu")(encoded)
    decoded = Dense(units=ENCODING_DIM*8, 
                    activation="relu")(bottleneck)
    output = Dense(units=input_output_dim, 
                    activation="linear")(decoded)
    autoencoder = Model(inputs=input_, outputs=output)
    return autoencoder

valid_rmse = []
folds = []
encoding_dim = []
epochs = []
last_train_rmse = []
last_val_rmse = []
idx = 0

for train_idx, valid_idx in kf.split(X_traindata):
  X_train, X_valid = X_traindata[train_idx], X_traindata[valid_idx]
  autoencoder = create_model()

  autoencoder.compile(optimizer='adam', loss='mse',
                        metrics=METRICS)
  history = autoencoder.fit(X_train, X_train,
                  batch_size=BATCH_SIZE,
                  validation_data=(X_valid, X_valid),
                  validation_steps=len(X_valid)//BATCH_SIZE,
                  epochs=EPOCHS)
  print(f'fold{idx}')
  autoencoder.save(os.path.join(DATADIR, 'weights', f'hidden{ENCODING_DIM}', 'fold'+str(idx)+'_lastEpoch.h5'))
  train_rmse = history.history['rms'][-1]
  val_rmse = history.history['val_rms'][-1]

  valid_rmse.append(history.history['val_rms'])
  folds.append([idx] * EPOCHS)
  encoding_dim.append([ENCODING_DIM] * EPOCHS)
  epochs.append([i for i in range(EPOCHS)])

  last_train_rmse.append(train_rmse)
  last_val_rmse.append(val_rmse)

  idx+=1
  plt.plot(history.history['val_rms'], label='val'+str(idx))
  plt.title(f'J = {ENCODING_DIM}')
  plt.ylabel('RMSE')
  plt.xlabel('Epochs')

valid_rmse = np.concatenate(valid_rmse)
encoding_dim = np.concatenate(encoding_dim)
folds = np.concatenate(folds)
epochs = np.concatenate(epochs)

data_tuples = list(zip(encoding_dim, folds, epochs, valid_rmse))
rmse_df9 = pd.DataFrame(data_tuples, columns=['Encoding_dim','Folds', 'Epochs', 'Valid_rmse'])
rmse_df9

ENCODING_DIM = 10 #Desired Dimension

def create_model(input_output_dim=160):
    input_ = Input(shape=(input_output_dim,))
    encoded = Dense(units=ENCODING_DIM*8, activation="relu")(input_)
    bottleneck = Dense(units=ENCODING_DIM, 
                       activation="relu")(encoded)
    decoded = Dense(units=ENCODING_DIM*8, 
                    activation="relu")(bottleneck)
    output = Dense(units=input_output_dim, 
                    activation="linear")(decoded)
    autoencoder = Model(inputs=input_, outputs=output)
    return autoencoder

valid_rmse = []
folds = []
encoding_dim = []
epochs = []
last_train_rmse = []
last_val_rmse = []
idx = 0

for train_idx, valid_idx in kf.split(X_traindata):
  X_train, X_valid = X_traindata[train_idx], X_traindata[valid_idx]
  autoencoder = create_model()

  autoencoder.compile(optimizer='adam', loss='mse',
                        metrics=METRICS)
  history = autoencoder.fit(X_train, X_train,
                  batch_size=BATCH_SIZE,
                  validation_data=(X_valid, X_valid),
                  validation_steps=len(X_valid)//BATCH_SIZE,
                  epochs=EPOCHS)
  print(f'fold{idx}')
  autoencoder.save(os.path.join(DATADIR, 'weights', f'hidden{ENCODING_DIM}', 'fold'+str(idx)+'_lastEpoch.h5'))
  train_rmse = history.history['rms'][-1]
  val_rmse = history.history['val_rms'][-1]

  valid_rmse.append(history.history['val_rms'])
  folds.append([idx] * EPOCHS)
  encoding_dim.append([ENCODING_DIM] * EPOCHS)
  epochs.append([i for i in range(EPOCHS)])

  last_train_rmse.append(train_rmse)
  last_val_rmse.append(val_rmse)

  idx+=1
  plt.plot(history.history['val_rms'], label='val'+str(idx))
  plt.title(f'J = {ENCODING_DIM}')
  plt.ylabel('RMSE')
  plt.xlabel('Epochs')

valid_rmse = np.concatenate(valid_rmse)
encoding_dim = np.concatenate(encoding_dim)
folds = np.concatenate(folds)
epochs = np.concatenate(epochs)

data_tuples = list(zip(encoding_dim, folds, epochs, valid_rmse))
rmse_df10 = pd.DataFrame(data_tuples, columns=['Encoding_dim','Folds', 'Epochs', 'Valid_rmse'])

rmse_df = pd.concat([rmse_df1, rmse_df2, rmse_df3, rmse_df4, rmse_df5, rmse_df6, rmse_df7, rmse_df8, rmse_df9, rmse_df10,])

rmse_re_df = rmse_df.reset_index()

sns.set(rc = {'figure.figsize':(15,8)})
sns.set_style("white")

p = sns.lineplot(data=rmse_re_df, x="Epochs", y="Valid_rmse", hue='Encoding_dim', palette='crest')
p.set_xlabel("Epochs", fontsize = 15)
p.set_ylabel("RMSE", fontsize = 15)
plt.legend([1,2,3,4,5,6,7,8,9,10], loc='upper right', title="Encoding dim ($\it{J}$)")
plt.ylim(0.7, 0.9)
plt.tight_layout()

# inferene with trained model
from keras.models import load_model

reduced_X_train1 = np.zeros((10, X_traindata.shape[0], 1))
for idx in range(0, 10):
  autoencoder = load_model(os.path.join(DATADIR, 'weights', f'hidden1', 'fold'+str(idx)+'_lastEpoch.h5'))
  encoder = Model(autoencoder.input, autoencoder.layers[-3].output)
  y = encoder.predict(X_traindata)
  reduced_X_train1[idx,:,:] = y
reduced_X_train1 = np.mean(reduced_X_train1, axis=0)

reduced_X_train2 = np.zeros((10, X_traindata.shape[0], 2))
for idx in range(0, 10):
  autoencoder = load_model(os.path.join(DATADIR, 'weights', f'hidden2', 'fold'+str(idx)+'_lastEpoch.h5'))
  encoder = Model(autoencoder.input, autoencoder.layers[-3].output)
  y = encoder.predict(X_traindata)
  reduced_X_train2[idx,:,:] = y
reduced_X_train2 = np.mean(reduced_X_train2, axis=0)

reduced_X_train3 = np.zeros((10, X_traindata.shape[0], 3))
for idx in range(0, 10):
  autoencoder = load_model(os.path.join(DATADIR, 'weights', f'hidden3', 'fold'+str(idx)+'_lastEpoch.h5'))
  encoder = Model(autoencoder.input, autoencoder.layers[-3].output)
  y = encoder.predict(X_traindata)
  reduced_X_train3[idx,:,:] = y
reduced_X_train3 = np.mean(reduced_X_train3, axis=0)

reduced_X_train4 = np.zeros((10, X_traindata.shape[0], 4))
for idx in range(0, 10):
  autoencoder = load_model(os.path.join(DATADIR, 'weights', f'hidden4', 'fold'+str(idx)+'_lastEpoch.h5'))
  encoder = Model(autoencoder.input, autoencoder.layers[-3].output)
  y = encoder.predict(X_traindata)
  reduced_X_train4[idx,:,:] = y
reduced_X_train4 = np.mean(reduced_X_train4, axis=0)

reduced_X_train5 = np.zeros((10, X_traindata.shape[0], 5))
for idx in range(0, 10):
  autoencoder = load_model(os.path.join(DATADIR, 'weights', f'hidden5', 'fold'+str(idx)+'_lastEpoch.h5'))
  encoder = Model(autoencoder.input, autoencoder.layers[-3].output)
  y = encoder.predict(X_traindata)
  reduced_X_train5[idx,:,:] = y
reduced_X_train5 = np.mean(reduced_X_train5, axis=0)

reduced_X_train6 = np.zeros((10, X_traindata.shape[0], 6))
for idx in range(0, 10):
  autoencoder = load_model(os.path.join(DATADIR, 'weights', f'hidden6', 'fold'+str(idx)+'_lastEpoch.h5'))
  encoder = Model(autoencoder.input, autoencoder.layers[-3].output)
  y = encoder.predict(X_traindata)
  reduced_X_train6[idx,:,:] = y
reduced_X_train6 = np.mean(reduced_X_train6, axis=0)

reduced_X_train7 = np.zeros((10, X_traindata.shape[0], 7))
for idx in range(0, 10):
  autoencoder = load_model(os.path.join(DATADIR, 'weights', f'hidden7', 'fold'+str(idx)+'_lastEpoch.h5'))
  encoder = Model(autoencoder.input, autoencoder.layers[-3].output)
  y = encoder.predict(X_traindata)
  reduced_X_train7[idx,:,:] = y
reduced_X_train7 = np.mean(reduced_X_train7, axis=0)

reduced_X_train8 = np.zeros((10, X_traindata.shape[0], 8))
for idx in range(0, 10):
  autoencoder = load_model(os.path.join(DATADIR, 'weights', f'hidden8', 'fold'+str(idx)+'_lastEpoch.h5'))
  encoder = Model(autoencoder.input, autoencoder.layers[-3].output)
  y = encoder.predict(X_traindata)
  reduced_X_train8[idx,:,:] = y
reduced_X_train8 = np.mean(reduced_X_train8, axis=0)

reduced_X_train9 = np.zeros((10, X_traindata.shape[0], 9))
for idx in range(0, 10):
  autoencoder = load_model(os.path.join(DATADIR, 'weights', f'hidden9', 'fold'+str(idx)+'_lastEpoch.h5'))
  encoder = Model(autoencoder.input, autoencoder.layers[-3].output)
  y = encoder.predict(X_traindata)
  reduced_X_train9[idx,:,:] = y
reduced_X_train9 = np.mean(reduced_X_train9, axis=0)

reduced_X_train10 = np.zeros((10, X_traindata.shape[0], 10))
for idx in range(0, 10):
  autoencoder = load_model(os.path.join(DATADIR, 'weights', f'hidden10', 'fold'+str(idx)+'_lastEpoch.h5'))
  encoder = Model(autoencoder.input, autoencoder.layers[-3].output)
  y = encoder.predict(X_traindata)
  reduced_X_train10[idx,:,:] = y
reduced_X_train10 = np.mean(reduced_X_train10, axis=0)

"""#5. K means clustering"""
##5.1. with raw data

import sklearn
from sklearn import metrics

X = train_scaled_df
inertia=[]
for n in range (1,11):
    model=KMeans(n_clusters=n, random_state=SEED, algorithm='full')
    model.fit(X)
    inertia.append(model.inertia_)

plt.figure(figsize=(8,6))
sns.set_style('whitegrid')
plt.plot(list(range(1,11)), inertia, linewidth=2, markersize=12, color='royalblue', marker='o',markerfacecolor='m', markeredgecolor='m')
plt.xlabel('Number of Clusters',fontsize=15)
plt.ylabel('Inertia',fontsize=15)
plt.title('Inertia vs. Number of Clusters',fontsize=18)
plt.show()

silhouette=[]
ch = []
for n in range (2,11):
    model=KMeans(n_clusters=n, random_state=SEED, algorithm='full')
    clusters = model.fit_predict(X)
    silhouette.append(metrics.silhouette_score(X, clusters,))
    ch.append(metrics.calinski_harabasz_score(X, clusters,))

##5.2. after PCA
X = pca_array
inertia=[]
for n in range (1,11):
    model=KMeans(n_clusters=n, random_state=SEED, algorithm='full')
    model.fit(X)
    inertia.append(model.inertia_)

plt.figure(figsize=(8,6))
sns.set_style('whitegrid')
plt.plot(list(range(1,11)), inertia, linewidth=2, markersize=12, color='royalblue', marker='o',markerfacecolor='m', markeredgecolor='m')
plt.xlabel('Number of Clusters',fontsize=15)
plt.ylabel('Inertia',fontsize=15)
plt.title('Inertia vs. Number of Clusters',fontsize=18)
plt.show()

silhouette=[]
ch = []
for n in range (2,11):
    model=KMeans(n_clusters=n, random_state=SEED, algorithm='full')
    clusters = model.fit_predict(X)
    silhouette.append(metrics.silhouette_score(X, clusters,))
    ch.append(metrics.calinski_harabasz_score(X, clusters,))

##5.3. after autoencoder
X = reduced_X_train2
inertia=[]
for n in range (1,11):
    model=KMeans(n_clusters=n, random_state=SEED, algorithm='full')
    model.fit(X)
    inertia.append(model.inertia_)

plt.figure(figsize=(8,6))
sns.set_style('whitegrid')
ax = plt.plot(list(range(1,11)), inertia, linewidth=2, markersize=8, color='royalblue', marker='o',markerfacecolor='blue', markeredgecolor='blue')
plt.xlabel('Number of clusters',fontsize=12)
plt.ylabel('WCSS',fontsize=12)
plt.grid(False)
plt.show()

X = reduced_X_train1
silhouette=[]
ch = []
for n in range (2,11):
    model=KMeans(n_clusters=n, random_state=SEED, algorithm='full')
    clusters = model.fit_predict(X)
    silhouette.append(metrics.silhouette_score(X, clusters,))
    ch.append(metrics.calinski_harabasz_score(X, clusters,))

X = reduced_X_train2
silhouette=[]
ch = []
for n in range (2,11):
    model=KMeans(n_clusters=n, random_state=SEED, algorithm='full')
    clusters = model.fit_predict(X)
    silhouette.append(metrics.silhouette_score(X, clusters,))
    ch.append(metrics.calinski_harabasz_score(X, clusters,))

X = reduced_X_train3
silhouette=[]
ch = []
for n in range (2,11):
    model=KMeans(n_clusters=n, random_state=SEED, algorithm='full')
    clusters = model.fit_predict(X)
    silhouette.append(metrics.silhouette_score(X, clusters,))
    ch.append(metrics.calinski_harabasz_score(X, clusters,))

X = reduced_X_train4
silhouette=[]
ch = []
for n in range (2,11):
    model=KMeans(n_clusters=n, random_state=SEED, algorithm='full')
    clusters = model.fit_predict(X)
    silhouette.append(metrics.silhouette_score(X, clusters,))
    ch.append(metrics.calinski_harabasz_score(X, clusters,))

X = reduced_X_train5
silhouette=[]
ch = []
for n in range (2,11):
    model=KMeans(n_clusters=n, random_state=SEED, algorithm='full')
    clusters = model.fit_predict(X)
    silhouette.append(metrics.silhouette_score(X, clusters,))
    ch.append(metrics.calinski_harabasz_score(X, clusters,))

X = reduced_X_train6
silhouette=[]
ch = []
for n in range (2,11):
    model=KMeans(n_clusters=n, random_state=SEED, algorithm='full')
    clusters = model.fit_predict(X)
    silhouette.append(metrics.silhouette_score(X, clusters,))
    ch.append(metrics.calinski_harabasz_score(X, clusters,))

X = reduced_X_train7
silhouette=[]
ch = []
for n in range (2,11):
    model=KMeans(n_clusters=n, random_state=SEED, algorithm='full')
    clusters = model.fit_predict(X)
    silhouette.append(metrics.silhouette_score(X, clusters,))
    ch.append(metrics.calinski_harabasz_score(X, clusters,))

X = reduced_X_train8
silhouette=[]
ch = []
for n in range (2,11):
    model=KMeans(n_clusters=n, random_state=SEED, algorithm='full')
    clusters = model.fit_predict(X)
    silhouette.append(metrics.silhouette_score(X, clusters,))
    ch.append(metrics.calinski_harabasz_score(X, clusters,))

X = reduced_X_train9
silhouette=[]
ch = []
for n in range (2,11):
    model=KMeans(n_clusters=n, random_state=SEED, algorithm='full')
    clusters = model.fit_predict(X)
    silhouette.append(metrics.silhouette_score(X, clusters,))
    ch.append(metrics.calinski_harabasz_score(X, clusters,))

X = reduced_X_train10
silhouette=[]
ch = []
for n in range (2,11):
    model=KMeans(n_clusters=n, random_state=SEED, algorithm='full')
    clusters = model.fit_predict(X)
    silhouette.append(metrics.silhouette_score(X, clusters,))
    ch.append(metrics.calinski_harabasz_score(X, clusters,))

"""#6. Model decision"""

X = reduced_X_train2
model=KMeans(n_clusters=n, random_state=SEED, algorithm='full')
model.fit(X)
labels=model.labels_
centers=model.cluster_centers_

clusters = model.predict(X)
cluster_df=train_scaled_df.copy()
cluster_df['Cluster'] = clusters

res_id = poorsleep_df['res_id']
res_id[:3]

quant_score = poorsleep_df[['PSQI_global',
                           'NQ_score',
                           'NQ_balance',
                           'NQ_diversity',
                           'NQ_moderation',
                           'NQ_behavior',
                           'GSRS_global',
                           'berlin_risk',]]

clustered_df = pd.concat([res_id, cluster_df, quant_score], axis=1, join='inner')

"""#7. Model inference"""

X_testdata = np.array(test_scaled_df)

reduced_X_test = np.zeros((10, X_testdata.shape[0], 2))
for idx in range(0, 10):
  autoencoder = load_model(os.path.join(DATADIR, 'weights', 'hidden2', 'fold'+str(idx)+'_lastEpoch.h5'))
  encoder = Model(autoencoder.input, autoencoder.layers[-3].output)
  y = encoder.predict(X_testdata)
  reduced_X_test[idx,:,:] = y

reduced_X_test = np.mean(reduced_X_test, axis=0)
reduced_X_test.shape

test_clusters = model.predict(reduced_X_test)
test_cluster_df=test_scaled_df.copy()
test_cluster_df['Cluster'] = test_clusters

res_id = poorsleep_df['res_id']
res_id[:3]

test_clustered_df = pd.concat([res_id, test_cluster_df, quant_score], axis=1, join='inner')
test_clustered_df['bmi'] = poorsleep_df.apply(lambda x: bmi_calculation(x['cm'], x['kg'],), axis=1)

"""#8. Internal cluster stats"""

sil_df = pd.read_csv(os.path.join(DATADIR, 'output', 'silhouette.csv'))
calinski_df = pd.read_csv(os.path.join(DATADIR, 'output', 'calinski.csv'))

p = sns.lineplot(data=sil_df, x='Cluster', y='Silhouette coefficient', hue='Method', palette='Set2') 
p.set_xlabel("Number of clusters", fontsize = 15)
p.set_ylabel('Silhouette coefficient', fontsize = 15)
plt.legend( loc='upper right', title="Methods")
plt.ylim(-0.1, 1.0)
plt.tight_layout()

p = sns.lineplot(data=calinski_df, x='Cluster', y='Calinski-Harabasz score', hue='Method', palette='Set2')
p.set_xlabel("Number of clusters", fontsize = 15)
p.set_ylabel('Calinski-Harabasz index', fontsize = 15)
plt.legend( loc='upper right', title='Methods')
plt.tight_layout()
