# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 23:12:09 2022

@author: beara
"""
import os
import pandas as pd
import numpy as np
import h5py
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from keras.models import load_model
from model import data_generator, nBatches


def load_encoder(
        workingDir = r"C:\Users\beara\Desktop\CS598\Project",
        modelDir = 'output'
        ):
    # Importing the pretrained autoencoder
    ae = load_model(os.path.join(workingDir, modelDir, 'auto_encoder.hdf5'))
    ae_encoder = ae.layers[1]
    
    return ae_encoder
    

def load_data_generator(seed = 1234,
                        workingDir = r"C:\Users\beara\Desktop\CS598\Project",
                        inputDir = 'data',
                        outputDir = 'output',
                        batch_size = 300):

    # Importing and splitting the sparsified syndromic records
    records = load_npz(os.path.join(workingDir, inputDir,'sparse_records.npz'))
    train_indices, test_indices = train_test_split(range(records.shape[0]), random_state=seed)
    train_recs = records[train_indices]
    test_recs  = records[test_indices]
    

    # Importing the text files
    sents = h5py.File(os.path.join(workingDir, inputDir,'word_sents.hdf5'), mode='r')
    train_sents = sents['X_train'].__array__()
    y_train = sents['y_train'].__array__()
    test_sents = sents['X_test'].__array__()
    y_test = sents['y_test'].__array__()
    
    # Importing the character lookup dictionary
    vocab_df = pd.read_csv(os.path.join(workingDir, inputDir,'word_dict.csv'))
    vocab = dict(zip(vocab_df['word'], vocab_df['value']))
   

    vocab_size = len(vocab.keys()) + 1
    
    # Setting up the data generators
    train_gen = data_generator(train_recs, train_sents, y_train,
                                 vocab_size=vocab_size,
                                 batch_size=batch_size)
    test_gen = data_generator(test_recs, test_sents, y_test,
                                vocab_size=vocab_size,
                                batch_size=batch_size)
    
    # return 
    sparse_size = records.shape[1]
    vocab_size = len(vocab.keys()) + 1
    max_length = train_sents.shape[1]
    
    
    n_train = len(y_train)
    n_test = len(y_test)
    train_steps = nBatches(n_train, batch_size)
    test_steps = nBatches(n_test, batch_size)
    

    return (train_gen, test_gen, train_steps, test_steps, 
            sparse_size, vocab_size, max_length,
            train_indices, test_indices,
            records, vocab)




def load_raw_complaints(workingDir, inputDir, seed=1234):
    # Reading in the data
    chiefComplaintColumn = 'chief complaint'
    variableColumns = ['age', 'GENDER', 'DIAGNOSIS']
    
    usecols = variableColumns + [chiefComplaintColumn]
    data = pd.read_csv(os.path.join(workingDir, inputDir,'parsed_data_all.csv'),
                          usecols=usecols)
    # delete rows with na in complaints
    data = data.dropna(subset=[chiefComplaintColumn], how='any')
    # use 'unknown' for na
    data = data.replace({np.nan: 'unknown'})
    data = data.reset_index(drop=True)
    
    return data