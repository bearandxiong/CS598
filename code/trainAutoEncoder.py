# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:19:59 2022

@author: beara
"""
import os
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keract import get_activations
from model import autoencoder
#import matplotlib.pyplot as plt
import seaborn as sns


workingDir = r"C:\Users\beara\Desktop\CS598\Project"
inputDir = 'data'
outputDir = 'output'
nEpochs = 100
batchSize = 1000


"""
Training Auto Encoder
"""

# Importing the data
records = load_npz(os.path.join(workingDir, inputDir,'sparse_records.npz'))
X_train, X_test = train_test_split(records, random_state=1234)

# Setting some global parameters
sparse_dim = records.shape[1]
embedding_dim = 128

# Training the model and loading the one with the lowest validation loss
X_train, X_test = X_train.todense(), X_test.todense()

mod = autoencoder(sparse_dim, embedding_dim)
mod.compile(optimizer='adam', loss='binary_crossentropy')



checkpointer = ModelCheckpoint(filepath=os.path.join(workingDir, outputDir, 'auto_encoder.hdf5'), 
                               save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', mode='auto', 
                               patience=5, restore_best_weights=True)

mod.fit(x=X_train, y=X_train, epochs=nEpochs, batch_size=batchSize, shuffle=True,
                validation_data=(X_test, X_test),
                callbacks=[checkpointer, early_stopping])


"""
Summarize the Auto Encoder
"""
mod.summary()


# =============================================
# Visualize input-->reconstruction
# =============================================
input_sample = X_train[-5000:]
output_sample = mod.predict(input_sample)

sns.heatmap(input_sample)


# =============================================
# Visualize encoded state with Keract
# =============================================
activations = get_activations(mod, input_sample)
sns.heatmap(activations['encoder'])
