# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:54:54 2022

@author: beara
"""
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import TextVectorization
import string
import re
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras import backend as K
# Model constants.

workingDir = r"C:\Users\beara\Desktop\CS598\Project"
inputDir = 'data'
outputDir = 'output'
max_features = 20000
embedding_dim = 128
epochs = 50
seed = 1234
batch_size = 300
sequence_length = 18

'''
prepare training data
'''
data = pd.read_csv(os.path.join(workingDir, inputDir, 'sents_icd9.csv'))
data['doc'] = [doc.replace('startingtag','').replace('endingtag','').strip() for doc in data['doc']]


'''
delete duplicated doc???
'''
#data = data.drop_duplicates(subset='doc')


# map categorical icd9 to numetic
# icd9 = data['icd9'].drop_duplicates().tolist()
# icd9_mapping = dict(zip(icd9,np.arange(len(icd9))))
# data['icd9']=data['icd9'].map(icd9_mapping)

encoder = LabelEncoder()
encoder.fit(data['icd9'])
encoded_Y = encoder.transform(data['icd9'])
dummy_y = np_utils.to_categorical(encoded_Y)


n_train = round(data.shape[0]*0.8)
n_test = data.shape[0] - n_train

x_train = data.head(n_train).doc
x_test = data.tail(n_test).doc
y_train = dummy_y[:n_train]
y_test = dummy_y[-n_test:]

nOutput = len(data['icd9'].drop_duplicates().tolist())


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)
# initialize the vectorization layer
vectorize_layer.adapt(data.doc)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label



# Vectorize the data.
train_ds = vectorize_text(x_train,  y_train)
test_ds  = vectorize_text(x_test,  y_test)


'''
build the model (after vectorize_layer)
'''
# A integer input for vocab indices.
inputs = tf.keras.Input(shape=(None,), dtype="int64")

# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
x = layers.Embedding(max_features, embedding_dim, name='embedding')(inputs)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.Conv1D(embedding_dim, 7, padding="valid", activation="relu", strides=1, name = 'conv1')(x)
x = layers.Conv1D(embedding_dim, 7, padding="valid", activation="relu", strides=1, name = 'conv2')(x)
x = layers.GlobalMaxPooling1D(name = 'pooling')(x)

# We add a vanilla hidden layer:
x = layers.Dense(embedding_dim, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
# 5808 is the number of unique icd9 codes
predictions = layers.Dense(nOutput, activation="softmax", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="categorical_crossentropy", optimizer="adam")

'''
train the model
'''
checkpointer = ModelCheckpoint(filepath=os.path.join(workingDir, outputDir,'trainedClassifier.hdf5'),
                               save_best_only=True,verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Fit the model using the train and test datasets.
model.fit(x = train_ds[0], y=train_ds[1], validation_data=test_ds, epochs=epochs,
          callbacks=[checkpointer, early_stopping])





# def f1(y_true, y_pred):    
#     def recall_m(y_true, y_pred):
#         TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
#         recall = TP / (Positives+K.epsilon())    
#         return recall 

#     def precision_m(y_true, y_pred):
#         TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
#         precision = TP / (Pred_Positives+K.epsilon())
#         return precision 
    
#     precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
#     f1= 2*((precision*recall)/(precision+recall+K.epsilon()))
#     return f1, precision, recall

'''
F1, precision and recall of the trained model
'''
nTotal = min(100000, train_ds[1].shape[0])
random_indices = np.random.choice(train_ds[1].shape[0], size=nTotal, replace=False)


truth = train_ds[1][random_indices]
pred = model.predict(np.array(train_ds[0])[random_indices])
#f1(truth, pred)

# use class labels instead
class_labels_true = np.argmax(truth, axis=1) 
class_labels_pred = np.argmax(pred, axis=1) 
print('f1: ',f1_score(class_labels_true, class_labels_pred, average='weighted'))
print('precision: ',precision_score(class_labels_true, class_labels_pred, average='weighted'))
print('recall: ', recall_score(class_labels_true, class_labels_pred, average='weighted'))
