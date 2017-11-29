""" Ejemplo de modelo con KERAS """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping

# Set seed
seed = 7
np.random.seed(seed)

#Read & edit the csv
data = pd.read_csv("Melbourne_housing_FULL.csv")
data.fillna(0, inplace=True)

def regionToInt(region):
    if region != 0:
        return {
            'Southern Metropolitan': 1,
            'Northern Metropolitan': 2,
            'Western Metropolitan': 3,
            'Eastern Metropolitan': 4,
            'South-Eastern Metropolitan': 5,
            'Eastern Victoria': 6,
            'Northern Victoria': 7,
            'Western Victoria': 8
        }[region]

data.Regionname = data.Regionname.apply(regionToInt)

def MethodToInt(method):
    if method != 0:
        return {
            'S': 1,
            'SP': 2,
            'PI': 3,
            'VB': 4,
            'SN': 5,
            'PN': 6,
            'SA': 7,
            'W': 8,
            'SS': 9
        }[method]

data.Method = data.Method.apply(MethodToInt)

#Create predictors matrix and target
predictors = data.drop(['Suburb', 'Address', 'Method', 'SellerG', 'CouncilArea',
                        'Propertycount', 'Date', 'Type'], axis=1).as_matrix()

target = np.array(data['Price'])
target.astype(int)

#Variable n_cols for input_shape
n_cols = predictors.shape[1]

#Model nº1
model_1 = Sequential()
model_1.add(Dense(50, activation = 'relu', input_shape = (n_cols,)))
model_1.add(Dense(1))

#Model nº2
model_2 = Sequential()
model_2.add(Dense(100, activation = 'relu', input_shape = (n_cols,)))
model_2.add(Dense(50, activation = 'relu', input_shape = (n_cols,)))
model_2.add(Dense(1))

#Model nº3
model_3 = Sequential()

model_3.add(Dense(50, activation = 'relu', input_shape = (n_cols,)))
model_3.add(Dense(20, activation = 'relu', input_shape = (n_cols,)))
model_3.add(Dense(1))

# Compile the models
model_1.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model_2.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model_3.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#EarlyStopping variable
early_stopping_monitor = EarlyStopping(patience=3)

# Fiting/training the models
model_fit = model_1.fit(predictors, target, epochs=20, validation_split=0.4,
                        callbacks=[early_stopping_monitor])

model_fit_2 = model_2.fit(predictors, target, epochs=20, validation_split=0.4,
                        callbacks=[early_stopping_monitor])

model_fit_3 = model_3.fit(predictors, target, epochs=20, validation_split=0.4,
                        callbacks=[early_stopping_monitor])

#Ploting models accurcy
plt.plot(model_fit.history['val_acc'], 'r', model_fit_2.history['val_acc'], 'b',
        model_fit_3.history['val_acc'], 'y')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.show()
