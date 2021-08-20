import random
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.util import bigrams

import torch
from tqdm import tqdm
from collections import defaultdict
import regex as re
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
import warnings
warnings.filterwarnings("ignore")

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

def word2vec_train(lst, emb_dim = 150, seed = 42):
    """
    train a word2vec mode
    args: lst(list of string): sentences
          emb_dim(int): word2vec embedding dimensions
          seed(int): seed for word2vec
    return: word2vec model
    """
    tokenizer = RegexpTokenizer(r'\w+')
    sentences = []
    for i in lst:
        sentences.append([x.lower() for x in tokenizer.tokenize(str(i))])
    w2v = Word2Vec(sentences, vector_size=emb_dim, min_count=1, seed=seed)
    return w2v

def get_sentence_emb(sentence, w2v):
    """
    get a sentence embedding vector
    *automatic initial random value to the new word
    args: sentence(string): sentence of log message
          w2v: word2vec model
    return: sen_emb(list of int): vector for the sentence
    """
    tokenizer = RegexpTokenizer(r'\w+')
    lst = []
    tokens = [x.lower() for x in tokenizer.tokenize(str(sentence))]
    if tokens == []:
        tokens.append('EmptyParametersTokens')
    for i in range(len(tokens)):
        words = list(w2v.wv.vocab.keys())
        if tokens[i] in words:
            lst.append(w2v[tokens[i]])
        else:
            w2v.build_vocab([[tokens[i]]], update = True)
            w2v.train([tokens[i]], epochs=1, total_examples=len([tokens[i]]))
            lst.append(w2v[tokens[i]])
    drop = 1
    if len(np.array(lst).shape) >= 2:
        sen_emb = np.mean(np.array(lst), axis=0)
        if len(np.array(lst)) >= 5:
            drop = 0
    else:
        sen_emb = np.array(lst)
    return list(sen_emb), drop