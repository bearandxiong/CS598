# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:12:33 2022

@author: beara
"""

import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer

NUMBER_OF_NGRAMS = 4


def find_ngrams(_input, n):
    if type(_input) == str:
        _input = _input.strip().split()
    out =  list(zip(*[_input[i:] for i in range(n)]))
    return [' '.join(item) for item in out]


def ngrams(sent, max_n=NUMBER_OF_NGRAMS):
    out = []
    for n in range(1, max_n + 1):
        out += find_ngrams(sent, n)
    return out


def cider(ref, test, vecs, n=NUMBER_OF_NGRAMS):
    '''
    cider(output.loc[0, 'chief complaint'], output.loc[0, 'greedy'], vecs)
    '''
    max_length = np.min([len(test.split()), len(ref.split())])
    to_use = np.intersect1d(range(max_length), range(n))
    tests = [vecs.transform([test]).toarray() for i in to_use]
    refs = [vecs.transform([ref]).toarray() for i in to_use]
    stat = np.mean([1 - cosine(tests[i], refs[i]) for i in to_use])
    return stat

def es(ref, test, vocab):

    test_split = [word for word in test.split() if word in list(vocab.keys())]
    ref_split = [word for word in ref.split() if word in list(vocab.keys())]
    
    to_use = len(vocab)
    test_vecs, ref_vecs = [0] * to_use, [0] * to_use
    
    for term in test_split:
        try:
            test_vecs[vocab[term]] = 1
        except:
            continue
    
    for term in ref_split:
        try:
            ref_vecs[vocab[term]] = 1
        except:
            continue    
    stat = 1 - cosine(test_vecs, ref_vecs)
    return stat

def ppv(ref, test, n=NUMBER_OF_NGRAMS):
    test = test.split()
    ref = ref.split()
    n = np.min([len(test), len(ref), n])
    test_grams = np.unique(ngrams(test, n))
    ref_grams = np.unique(ngrams(ref, n))
    return np.sum([gram in ref_grams for gram in test_grams]) / len(test_grams)

def sens(ref, test, n=NUMBER_OF_NGRAMS):
    test = test.split()
    ref = ref.split()
    n = np.min([len(test), len(ref), n])
    test_grams = np.unique(ngrams(test, n))
    ref_grams = np.unique(ngrams(ref, n))
    return np.sum([gram in test_grams for gram in ref_grams]) / len(ref_grams)

def f1(ppv_score, sens_score):
    if ppv_score ==0 and sens_score ==0:
        return 0
    else:
        return 2*(ppv_score* sens_score)/(ppv_score+ sens_score)


if __name__== '__main__':
    workingDir = r"C:\Users\beara\Desktop\CS598\Project"
    inputDir = 'data'
    outputDir = 'output'
    
    output = pd.read_excel(os.path.join(workingDir, outputDir, 'generated_text(epochs=100).xlsx'))
    vecs = CountVectorizer(binary=False, ngram_range=(1, 1),
                           token_pattern="(?u)\\b\\w+\\b",decode_error='ignore')
    vecs.fit(output['chief complaint'])
    vocab = vecs.vocabulary_
    
    statistics= []
    for method in ['greedy', 'sampling', 'beam']:
        ppv_stats = []
        sens_stats = []
        f1_stats = []
        cider_stats = []
        es_stats = []
        for i in range(output.shape[0]):
            if i% 10000 ==0 and i > 0:
                print(f"{method}: Finished {i} records generation, remaining {output.shape[0]-i} to work on...")
            ppv_score = ppv(output.loc[i, 'chief complaint'], str(output.loc[i, method]))
            sens_score = sens(output.loc[i, 'chief complaint'], str(output.loc[i, method]))
            ppv_stats.append(ppv_score)
            sens_stats.append(sens_score)
            f1_stats.append(f1(ppv_score, sens_score))
            cider_stats.append(cider(output.loc[i, 'chief complaint'],  str(output.loc[i, method]), vecs))
            es_stats.append(es(output.loc[i, 'chief complaint'], str(output.loc[i, method]), vocab))
        
        ppv_stats = np.array(ppv_stats)
        sens_stats = np.array(sens_stats)
        f1_stats = np.array(f1_stats)
        cider_stats = np.array(cider_stats)
        es_stats = np.array(es_stats)
        
        result = pd.DataFrame({'ppv':[np.nanmean(ppv_stats[np.where(ppv_stats>0)])],
                               'sens':[np.nanmean(sens_stats[np.where(sens_stats>0)])],
                               'f1':[np.nanmean(f1_stats[np.where(f1_stats>0)])],
                               'cider':[np.nanmean(cider_stats[np.where(cider_stats>0)])],
                               'es':[np.nanmean(es_stats[np.where(es_stats>0)])],
                               'method':[method]
                               })
        
        statistics.append(result)
    
    '''
    beam search with k= 2, 3, 4 candidates
    '''
    method = 'beam'
    for nCandidates in [2,3,4]:
        ppv_stats = []
        sens_stats = []
        f1_stats = []
        cider_stats = []
        es_stats = []
        for i in range(output.shape[0]):
            if i% 10000 ==0 and i > 0:
                print(f"{method}, k={nCandidates}: Finished {i} records generation, \
                      remaining {output.shape[0]-i} to work on...")
                
            pred = ' '.join(str(output.loc[i, method]).split('<&>')[:nCandidates])    
            
            ppv_score = ppv(output.loc[i, 'chief complaint'], pred)
            sens_score = sens(output.loc[i, 'chief complaint'], pred)
            ppv_stats.append(ppv_score)
            sens_stats.append(sens_score)
            f1_stats.append(f1(ppv_score, sens_score))
            cider_stats.append(cider(output.loc[i, 'chief complaint'], pred, vecs))
            es_stats.append(es(output.loc[i, 'chief complaint'], pred, vocab))
        
        ppv_stats = np.array(ppv_stats)
        sens_stats = np.array(sens_stats)
        f1_stats = np.array(f1_stats)
        cider_stats = np.array(cider_stats)
        es_stats = np.array(es_stats)
        
        result = pd.DataFrame({'ppv':[np.nanmean(ppv_stats[np.where(ppv_stats>0)])],
                               'sens':[np.nanmean(sens_stats[np.where(sens_stats>0)])],
                               'f1':[np.nanmean(f1_stats[np.where(f1_stats>0)])],
                               'cider':[np.nanmean(cider_stats[np.where(cider_stats>0)])],
                               'es':[np.nanmean(es_stats[np.where(es_stats>0)])],
                               'method':[method + " k=" + str(nCandidates)]
                               })
        statistics.append(result)
        
        
    # convert to pandas
    statistics = pd.concat(statistics, axis=0)
    statistics.to_excel(os.path.join(workingDir, outputDir, 'model_statistics.xlsx'), index=False)
        
        
        
        
