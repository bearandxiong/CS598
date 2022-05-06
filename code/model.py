# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:19:41 2022

@author: beara
"""

import itertools
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, Reshape
from keras import backend as K
from keras.models import load_model

from preProcessing import one_hot, ints_to_text


'''
---------------------------------------------------------------------------------------------
Auto Encoder model
---------------------------------------------------------------------------------------------
'''
def autoencoder(sparse_dim, embedding_dim):
    data = Input(shape=(sparse_dim,))
    encoder = Dense(embedding_dim, activation='relu', name='encoder')
    decoder = Dense(sparse_dim, activation='sigmoid', name='decoder')
    decodedData = decoder(encoder(data))
    return Model(data, decodedData)

'''
utilities for training
'''
def make_starter(max_length, indices=None):
    out = np.zeros([1, max_length])
    if indices is not None:
        for i, index in enumerate(indices):
            out[0][i] = index
    return out


def nBatches(n, batch_size):
    return int((n / batch_size) + (n % batch_size))


def data_generator(recs, sents, y, batch_size=200, vocab_size=50):
    n_batches = nBatches(len(y), batch_size)
    counter=0
    shuffle_index = np.arange(np.shape(y)[0])
    np.random.shuffle(shuffle_index)
    while True:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        rec_batch = recs[index_batch].toarray().astype(dtype=np.uint8)
        sent_batch = sents[index_batch].astype(dtype=np.uint32)
        y_batch = np.array([one_hot(sent, vocab_size).transpose() 
                            for sent in y[index_batch]], dtype=np.uint8)
        counter += 1
        yield([rec_batch, sent_batch], y_batch)
        if (counter < n_batches):
            np.random.shuffle(shuffle_index)
            counter=0

def top_n(data, n):
    ind = np.argpartition(data, -n)[-n:]
    return ind


'''
---------------------------------------------------------------------------------------------
main model class
---------------------------------------------------------------------------------------------
'''
class myModel(object):
    def __init__(self, sparse_size, vocab_size, max_length, embedding_size, hidden_size,
                 embeddings_dropout=0.2, recurrent_dropout=0.2):
        self.sparse_size = sparse_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.text_embedding = None
        
        # define layers
        self.input_embedding = Dense(units=embedding_size, name='input_embedding', trainable=True)
        self.text_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_size,
                                        embeddings_regularizer=None, mask_zero=True,
                                        name='text_embedding')
        self.rnn = LSTM(units=hidden_size, dropout=embeddings_dropout, recurrent_dropout=recurrent_dropout,
                        return_sequences=True, return_state=True, name='rnn')
        self.output_layer = Dense(units=vocab_size, activation='softmax', name='output_layer')
        
        self.record_lookup = None
        self.training_model = None
        self.inference_model = None
        

    def build_training_model(self, encoding_layer=None):
        
        if encoding_layer != None:
            ae_weights = encoding_layer.get_weights()
            self.record_embedding_layer = Dense(units=self.embedding_size,name='input_embedding',
                                                trainable=True, weights=ae_weights)

        # sparse record embedding
        data = Input(shape=(self.sparse_size,), name='sparse_record')
        embedded_data = self.input_embedding(data)
        reshaped_data = Reshape((1, self.embedding_size))(embedded_data)

        # text embedding
        input_text = Input(shape=(self.max_length,), name='text_input')
        text_embedding = self.text_embedding(input_text)

        # rnn
        batch_size = K.shape(data)[0]
        zero_state = [K.zeros((batch_size, self.hidden_size)),
                      K.zeros((batch_size, self.hidden_size))]

        # Running the record through the RNN first, and then the text
        rec_out, rec_h, rec_c = self.rnn(reshaped_data, initial_state=zero_state)
        rnn_output, _, _ = self.rnn(text_embedding, initial_state=[rec_h, rec_c])

        # Adding a dense layer with softmax for getting predictions
        inputs, output = [data, input_text], self.output_layer(rnn_output)

        # saving to class object
        self.record_lookup  = Model(data, [rec_h, rec_c])
        self.training_model = Model(inputs=inputs, outputs=output)
        
    """
    def load_training_model(self, mod_path):
        mod = load_model(mod_path)

        # load the layers
        self.rnn = [layer for layer in mod.layers if layer.name == 'rnn'][0]
        self.text_embedding = [layer for layer in mod.layers if layer.name == 'text_embedding'][0]
        self.input_embedding = [layer for layer in mod.layers if layer.name == 'input_embedding'][0]
        self.output_layer = [layer for layer in mod.layers if layer.name == 'output_layer'][0]
        # bulild the model
        self.build_inference_model()
    """
        

    # Builds the inference model for the captioner using the embedding
    # layer, RNN, and dense layers from the training model
    def build_inference_model(self):

        # Defining inputs for the states of the encoder model
        input_h = Input(shape=(self.hidden_size,))
        input_c = Input(shape=(self.hidden_size,))
        input_states = [input_h, input_c]

        # Defining an input for the text sequence
        input_text = Input(shape=(self.max_length,), name='text_input')
        text_embedding = self.text_embedding(input_text)

        # Running a step through the RNN
        rnn_output, output_h, output_c = self.rnn(text_embedding, initial_state=input_states)
        output_states = [output_h, output_c]

        # output layer
        output = self.output_layer(rnn_output)
        model = Model([input_text] + input_states, [output] + output_states)
        self.inference_model = model
        
        
    def beam_filter(self, seq, states, k=5):
        previous = seq[np.where(seq != 0)]
        probs, h, c = self.inference_model.predict([seq] + [states[0], states[1]])
        best_next = top_n(probs[0, len(previous)-1, :], k)
        best_seqs = np.array([np.concatenate([previous, [_next]]) for _next in best_next])
        beam_probs = np.array([probs[0,:, _next] for _next in best_next])
        # sum of log (= log of product) avoiding underflow
        scores = np.sum(np.log(beam_probs + 1e-10), axis=1)
        states = list(itertools.repeat([h, c], k))
        out_seqs = np.array([make_starter(self.max_length, seq) for seq in best_seqs])
        return {'seqs':out_seqs, 'states':states, 'scores':scores}
    
   
    def beamSearch(self, record, vocab, k=5):
        seed, end = vocab['startingtag'], vocab['endingtag']
        caption = make_starter(self.max_length, [seed])
        seed_length = np.sum(seed != 0)
        # Getting the embedded version of the record to pass to the RNN
        feed_states = self.record_lookup.predict(record.toarray())
        
        # Running the first beam with the seed sequence
        seed_beam = self.beam_filter(caption, feed_states, k=k)
        live_seqs = seed_beam['seqs']
        beam_states = seed_beam['states']
        dead_seqs = list([])
        search_end = self.max_length - 1

        # Running the main loop for the subsequent beams
        for i in range(seed_length, search_end):
            if live_seqs.shape[0]==0 or k==0:
                break
            beams = list([])
            # Running a beam for each of the candidate sequences
            for j, seq in enumerate(live_seqs):
                in_states = beam_states[j]
                beams.append(self.beam_filter(seq, in_states, k=k))

            # Getting the states, sequences, and scores from each beam
            beam_states = np.concatenate([beam['states'] for beam in beams], 0)
            beam_seqs = np.concatenate([beam['seqs'] for beam in beams], 0)
            beam_scores = np.concatenate([beam['scores'] for beam in beams], 0)

            # topk is the index of teh top n
            topk = top_n(beam_scores, n=k)
            live_seqs = beam_seqs[topk, :]
            beam_states = beam_states[topk, :]

            # Finding finished sequences and trimming the live ones
            any_dead = np.any(live_seqs == end, axis=2)
            if np.any(any_dead):
                where_dead = np.where(any_dead)[0]
                current_dead = live_seqs[where_dead, :]
                live_seqs = np.delete(live_seqs, where_dead, 0)
                [dead_seqs.append(seq) for seq in current_dead]
                k -= len(current_dead)

            # Adding any leftover live sequences to the dead ones
            if i == search_end:
                [dead_seqs.append(seq) for seq in live_seqs]

        # Returning the finished sequences
        dead_seqs = np.array(dead_seqs)
        out = np.array([ints_to_text(seq[0], vocab) for seq in dead_seqs])
        
        return out
    
   
    def randomSearch(self, record, vocab):
        seed, end = vocab['startingtag'], vocab['endingtag']
        caption = make_starter(self.max_length, [seed])
        seed_length = np.sum(seed != 0)
        # Getting the embedded version of the record to pass to the RNN
        feed_states = self.record_lookup.predict(record.toarray())
        
        for i in range(seed_length, self.max_length):
            probs, h, c = self.inference_model.predict([caption] + feed_states)
            current_probs = probs[0, i-1, :]
            melted_probs = np.log(current_probs)
            exp_melted = np.exp(melted_probs)
            new_probs = exp_melted / np.sum(exp_melted)
            best = np.random.choice(range(self.vocab_size), p=new_probs)
            caption[0, i] = best
            feed_states = [h, c]
            if best == end:
                break
        out = ints_to_text(caption[0], vocab)
        return out
        
   
    def greedySearch(self, record, vocab, k=5):
        seed, end = vocab['startingtag'], vocab['endingtag']
        caption = make_starter(self.max_length, [seed])
        seed_length = np.sum(seed != 0)
        # Getting the embedded version of the record to pass to the RNN
        feed_states = self.record_lookup.predict(record.toarray())
        
        for i in range(seed_length, self.max_length):
            probs, h, c = self.inference_model.predict([caption] + feed_states)
            best = np.argmax(probs[0, i-1, :])
            caption[0, i] = best
            feed_states = [h, c]
            if best == end:
                break
        out = ints_to_text(caption[0], vocab)
        return out
        
    
    def generateText(self, record, vocab, method='greedy', k=5):
        if method == 'greedy':
            return self.greedySearch(record, vocab, k)
        elif method == 'beam':
            return self.beamSearch(record, vocab, k)
        elif method == 'sampling':
            return self.randomSearch(record, vocab, k)
        else:
            raise NotImplementedError(f'Unknown sampling {method}\n\
                                      use from {"greedy", "beam","sampling"}')
        

