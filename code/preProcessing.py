# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 08:25:45 2022

@author: beara
"""

import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from scipy.sparse import save_npz
import h5py


def to_integer(tokens, vocab_dict):
    return np.array([vocab_dict[token] for token in tokens], dtype=np.uint32)


def get_key(value, dic, pad=0):
    if value != pad:
        return list(dic.keys())[list(dic.values()).index(value)]
    else:
        return 'pad'
    
    
def ints_to_text(values, dic):
    values = values[np.where(values != 0)[0]]
    values = values[1:-1]
    out = [get_key(val, dic) for val in values]
    return ' '.join(out)


def pad_integers(phrase, max_length):
    pad_size = max_length - len(phrase)
    if pad_size>=0:
        return np.concatenate((phrase, np.repeat(0, pad_size)))
    else:
        return phrase[:max_length]
    

def one_hot(indices, vocab_size):
    mat = np.zeros((vocab_size, indices.shape[0]), dtype=np.uint8)
    mat[indices, np.arange(mat.shape[1])] = 1
    mat[0, :] = 0
    return mat


def to_sparse(_input):
    vec = CountVectorizer(binary=True, ngram_range=(1, 1), token_pattern="(?u)\\b\\w+\\b")
    data = vec.fit_transform(_input)
    vocab = sorted(vec.vocabulary_.keys())
    return {'data':data, 'vocab':vocab}


def preProcessing(workingDir=r"C:\Users\beara\Desktop\CS598\Project",
                  inputDir='data', outputDir='data'):
    # Reading in the data
    chiefComplaintColumn = 'chief complaint'
    variableColumns = ['age', 'GENDER', 'DIAGNOSIS', 'ICD9_CODE']
    
    usecols = variableColumns + [chiefComplaintColumn]
    data = pd.read_csv(os.path.join(workingDir, inputDir,'parsed_data_all.csv'),
                          usecols=usecols)
    # delete rows with na in complaints
    data = data.dropna(subset=[chiefComplaintColumn], how='any')
    # use 'unknown' for na
    data = data.replace({np.nan: 'unknown'})

    sparseData = [to_sparse(data[col].astype(str)) for col in variableColumns]
    outputCSR = hstack([col['data'] for col in sparseData], format='csr')
    sparseVocab= [col['vocab'] for col in sparseData]
    sparseVocab = pd.Series([item for sublist in sparseVocab for item in sublist])
    
    # Writing the files to disk
    save_npz(os.path.join(workingDir, inputDir,'sparse_records.npz'), outputCSR)
    sparseVocab.to_csv(os.path.join(workingDir, inputDir,'sparse_vocab.csv'), index=False)


    """
    text to integer
    """
    # Prepping the text column
    codes = data['ICD9_CODE']
    text = data[chiefComplaintColumn].str.lower()
    text = ['startingtag ' + doc + ' endingtag' for doc in text]
    
    # First-pass vectorization to get the overall vocab
    text_vec = CountVectorizer(binary=False,
                               ngram_range=(1, 1),
                               token_pattern="(?u)\\b\\w+\\b",
                               decode_error='ignore')
    text_vec.fit(text)
    vocab = text_vec.vocabulary_
    
    # Adding 1 to each vocab index to allow for 0 masking
    # 0 is stopping word
    for word in vocab:
        vocab[word] += 1
    
    # Writing the vocabulary to disk
    vocab_df = pd.DataFrame.from_dict(vocab, orient='index')
    vocab_df['word'] = vocab_df.index
    vocab_df.columns = ['value', 'word']
    vocab_df.to_csv(os.path.join(workingDir, inputDir,'word_dict.csv'), index=False)
    
    # Weeding out docs longer than max_length (18 is the default)
    max_length = 18
    text_series = pd.Series(text)
    code_series = pd.Series(codes)
    
    doc_lengths = np.array([len(doc.split()) for doc in text_series])
    text_series = text_series.iloc[np.where(doc_lengths <= max_length)[0]]
    
    # Weeding out docs with tokens that CountVectorizer doesn't recognize;
    # mostly a redundancy check for funky characters.
    in_vocab = np.where([np.all([word in vocab.keys() for word in doc.split()]) for doc in text_series])
    good_docs = text_series.iloc[in_vocab]
    good_code = code_series.iloc[in_vocab]
    
    # Setting up the train-test splits
    n = good_docs.shape[0]
    train_indices, test_indices = train_test_split(range(n), random_state=1234)
    
    
    """
    save data to file in df5 format
    """
    
    # Preparing the HDF5 file to hold the output
    output = h5py.File(os.path.join(workingDir, inputDir,'word_sents.hdf5'), mode='w')
    
    # Running and saving the splits for the inputs; going with np.uin16
    # for the dtype since the vocab size is much smaller than before
    int_sents = np.array([pad_integers(to_integer(doc.split()[:-1], vocab), max_length) 
                          for doc in good_docs], dtype=np.uint16)
    output['X_train'] = int_sents[train_indices]
    output['X_test']  = int_sents[test_indices]
    
    # And doing the same for the outputs
    targets = np.array([pad_integers(to_integer(doc.split()[1:], vocab), max_length) 
                        for doc in good_docs], dtype=np.uint16)
    output['y_train'] = targets[train_indices]
    output['y_test'] = targets[test_indices]
    
    # Shutting down the HDF5 file
    output.close()
    
    
    '''
    saving notes to icd9 data, for classifier training
    '''
    docList = []
    codeList = []
    for doc, codes in zip(good_docs, good_code):
        codes = codes.split(';')
        docList += [doc]*len(codes)
        codeList += codes
    data = pd.DataFrame({'doc': docList, 'icd9':codeList})
    data.to_csv(os.path.join(workingDir, inputDir,'sents_icd9.csv'), index=False)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    