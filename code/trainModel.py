# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:53:43 2022

@author: beara
"""
import os
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from loadTrainingData import (load_encoder, 
                              load_data_generator, 
                              load_raw_complaints)
from model import myModel
from scipy.sparse import vstack

'''
---------------------------------------------------------------------------------------------
# Setting the parameters for training
---------------------------------------------------------------------------------------------
'''
workingDir = r"C:\Users\beara\Desktop\CS598\Project"
inputDir = 'data'
outputDir = 'output'
seed = 1234
hidden_size = 128
embedding_size = 128
batch_size = 300
epochs = 100


# load the encoder layer
ae_encoder = load_encoder(workingDir = workingDir, modelDir = outputDir)

# load model ready data generator etc.
train_gen, test_gen, train_steps, test_steps, \
    sparse_size, vocab_size, max_length, \
        train_indices, test_indices, \
            records, vocab=\
    load_data_generator(seed = seed, batch_size = batch_size,
                        workingDir = workingDir, 
                        inputDir = inputDir, outputDir = outputDir)



# define checkpint
checkpointer = ModelCheckpoint(filepath=os.path.join(workingDir, outputDir,'trainedModel.hdf5'),
                               save_best_only=True,verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)


# Setting up the NRC model
mod = myModel(embedding_size=embedding_size, sparse_size=sparse_size,
              hidden_size=hidden_size,vocab_size=vocab_size,max_length=max_length)


# Building and running the model in training mode
mod.build_training_model(encoding_layer=ae_encoder)
mod.training_model.compile(optimizer='adam',
                           loss='categorical_crossentropy')
mod.training_model.fit(train_gen, verbose=1, epochs=epochs, steps_per_epoch=train_steps,
                       validation_data=test_gen, validation_steps=test_steps,
                       callbacks=[checkpointer, early_stopping])
mod.build_inference_model()



"""
---------------------------------------------------------------------------------------------
# generate text by fitted model
---------------------------------------------------------------------------------------------
"""

raw_complaints = load_raw_complaints(workingDir, inputDir)

train_complaints, test_comclaints = raw_complaints.loc[train_indices], raw_complaints.loc[test_indices]
train_recs, test_recs = records[train_indices], records[test_indices]

# add split indicator
train_complaints['split'] = 'train'
test_comclaints['split'] = 'test'
raw_complaints = pd.concat([train_complaints, test_comclaints], axis=0)
# reorder the records
records = vstack((train_recs, test_recs))




'''
# too slow
beamResult = []
samplingResult = []
greedyResult = []

for i, _record in enumerate(records):
    if i% 100 ==0 and i > 0:
        print(f"Finished {i} records generation, remaining {records.shape[0]-i} to work on...")
    
    sampling_line_result = mod.generateText(_record, vocab, method = 'sampling')
    greedy_line_result = mod.generateText(_record, vocab, method = 'greedy')
    beam_line_result = mod.generateText(_record, vocab, method = 'beam')
    # list to string
    beam_line_result = "<&>".join(beam_line_result)
    
    beamResult.append(beam_line_result)
    samplingResult.append(sampling_line_result)
    greedyResult.append(greedy_line_result)

output = raw_complaints.copy()
output['greedy'] = greedyResult
output['sampling'] = samplingResult
output['beam'] = beamResult


# save the result to excel
output.to_excel(os.path.join(workingDir, outputDir, 'generated_text.xlsx'), index=False)



'''
# parallel loop

from joblib import Parallel, delayed, parallel_backend

def generate(i):
    if i% 100 ==0 and i > 0:
        print(f"Finished {i} records generation, remaining {records.shape[0]-i} to work on...")
    _record = records[i]
    sampling_line_result = mod.generateText(_record, vocab, method = 'sampling')
    greedy_line_result = mod.generateText(_record, vocab, method = 'greedy')
    beam_line_result = mod.generateText(_record, vocab, method = 'beam')
    # list to string
    beam_line_result = "<&>".join(beam_line_result)
    return sampling_line_result, greedy_line_result, beam_line_result
    
with parallel_backend('threading', n_jobs=8):    
    results = Parallel(n_jobs=8)(delayed(generate)(i) for i in range(records.shape[0]))

# unpack the results
sampling_result, greedy_result, beam_result =  list(zip(*results))


output = raw_complaints.copy()
output['greedy'] = greedy_result
output['sampling'] = sampling_result
output['beam'] = beam_result


# save the result to excel
output.to_excel(os.path.join(workingDir, outputDir, f'generated_text(epochs={epochs}).xlsx'), index=False)








