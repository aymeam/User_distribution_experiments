#from data_handler import *
import argparse
import keras
import numpy as np
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, LSTM
from keras.models import Sequential, Model
import pdb
import pickle
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.model_selection import KFold, StratifiedKFold
from keras.utils import np_utils
import codecs
import operator
import gensim, sklearn
from string import punctuation
from collections import defaultdict
import sys
import xgboost as xgb
import json
from nltk import tokenize as tokenize_nltk
from string import punctuation
from preprocess_twitter import tokenize as tokenizer_g
from gensim.parsing.preprocessing import STOPWORDS
import random
import pdb
import math
from keras import backend as K
from sklearn.metrics import roc_auc_score
import preprocessor as p
from collections import Counter
import os
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix 
from tensorflow.contrib import learn
from tflearn.data_utils import to_categorical, pad_sequences
from scipy import stats
import tflearn
import json
from nltk import tokenize as tokenize_nltk
TOKENIZER = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
import pandas as pd
import operator
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import random
from models import *
import keras
from sklearn.metrics import accuracy_score


SEED = 42
np.random.seed(SEED)

def mult(vector,entero):
    vector_nuevo = []
    for i in vector:
        vector_nuevo.append(round((100*i),2))
    return vector_nuevo

def from_python_to_latex(clases, experiment_name, models_names, precision,recall,fscore,
                         avg_precision,avg_recall,avg_fscore,avg_precisionM,avg_recallM,avg_fscoreM):
        with open(experiment_name +  '.tex', 'a') as output_file:
            output_file.write(" \\begin{table}"+'\n'+ " \centering" +'\n'+ " \caption{}" +'\n'+ " \label{Table-3}"+'\n'+
"{ \small" +'\n'+ " \\begin{tabular}{llllll}" +'\n'+ "Method&Class & Prec. & Rec. & F1. \\\ " +'\n'+ " \midrule" +'\n')
            for model in models_names:
                for i in range(len(precision)):
                    if i==0:
                        cadena = model+ "&"
                    else:
                        cadena = "&"
                    cadena += str(clases[i]) + "&" + str(precision[i]) + "&" +  str(recall[i]) + "&" +  str(fscore[i]) + " \\\ " + "\n"
                    output_file.write(cadena)
               
                output_file.write(" \\addlinespace " + "\n" )
               
                output_file.write("&" + "Micro" + "&" + str(avg_precision) + "&" +  str(avg_recall) + "&" +  str(avg_fscore) + " \\\ " + "\n")
                output_file.write("&" + "Macro" + "&" + str(avg_precisionM) + "&" +  str(avg_recallM) + "&" +  str(avg_fscoreM) + " \\\ " + "\n")
               
                output_file.write(" \\midrule " + "\n" )
            output_file.write("\\bottomrule" + "\n"+ "\end{tabular}" + "\n" + "}" + "\n"+ "\end{table}")

def save_object(obj, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)

def load_object(filename):
    with open(filename, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def batch_gen(X, batch_size):
    n_batches = X.shape[0]/float(batch_size)
    n_batches = int(math.ceil(n_batches))
    end = int(X.shape[0]/float(batch_size)) * batch_size
    n = 0
    for i in range(0,n_batches):
        if i < n_batches - 1: 
            batch = X[i*batch_size:(i+1) * batch_size, :]
            yield batch
        
        else:
            batch = X[end: , :]
            n += X[end:, :].shape[0]
            yield batch


def glove_tokenize(text):
    text = tokenizer_g(text)
    text = ''.join([c for c in text if c not in punctuation])
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return words


import io
import numpy as np
def load_vec(emb_path,lang, nmax=50000):
    if lang == 'man':
       file.write("149993 300" + '\n')
    vectors = []
    word2id = {}
    words=[".",","]
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            if word not in words:
#               file.write(word + " ")
              i = 0
#               for v in vect:
#                 #print(v)
#                 file.write(str(v) + " ")
#               file.write('\n')
            else:
              words.append(word)
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


#word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('../Vectors/glove.twitter.27B.200d.txt')


ita_embeddings, ita_id2word, ita_word2id = load_vec('../Vectors/wiki.multi.it.vec.txt','ita')
es_embeddings, es_id2word, es_word2id = load_vec('../Vectors/wiki.multi.es.vec.txt','es')
en_embeddings, en_id2word, en_word2id = load_vec('../Vectors/wiki.multi.en.vec.txt','es')

freq = defaultdict(int)
EMBEDDING_DIM = 200
NO_OF_FOLDS = 10
LEARN_EMBEDDINGS = True
SCALE_LOSS_FUN = None
TOKENIZER = glove_tokenize#tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize

def gradient_boosting_classifier(X_train, y_train):
    a, p, r, f1 = 0., 0., 0., 0.
    a1, p1, r1, f11 = 0., 0., 0., 0.
    pn, rn, fn = 0., 0., 0.
    clf = xgb.XGBClassifier(nthread=-1)  
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(model, testX, testY,flag):
    if flag=='binary':
        temp = model.predict(testX)
        y_pred = []
        for i in temp:
            if i >0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)  

        y_true = []
        for i in testY:
            if i >0.5:
                y_true.append(1)
            else:
                y_true.append(0) 
                
    elif flag == 'categorical':
        
        y_true = testY
        y_pred = model.predict(testX)
        
        #y_pred = np.argmax(y_pred, axis=1)
    
    else:
        y_true = testY 
        temp = model.predict(testX)
        y_pred_aux = temp
        y_pred=[]
        for i in y_pred_aux:
            if i == 2:
                y_pred.append(1)
            else:
                y_pred.append(i)
        
    precision = metrics.precision_score(y_true, y_pred, average=None)
    recall = metrics.recall_score(y_true, y_pred, average=None)
    f1_score = metrics.f1_score(y_true, y_pred, average=None)
    
    precisionw = metrics.precision_score(y_true, y_pred, average='weighted')
    recallw = metrics.recall_score(y_true, y_pred, average='weighted')
    f1_scorew = metrics.f1_score(y_true, y_pred, average='weighted')
    
    precisionm = metrics.precision_score(y_true, y_pred, average='macro')
    recallm = metrics.recall_score(y_true, y_pred, average='macro')
    f1_scorem = metrics.f1_score(y_true, y_pred, average='macro')
    
    print("Precision: " + str(precision) + "\n")
    print("Recall: " + str(recall) + "\n")
    print("f1_score: " + str(f1_score) + "\n")
    #print(confusion_matrix(y_true, y_pred))
    
    print(":: Classification Report")
    print(classification_report(y_true, y_pred))
    return precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem


def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)


def prod(list, num):
    res=[]
    for i in list:
        res.append(i/num)
    return res
def mult(list, num):
    res=[]
    for i in list:
        res.append(i*num)
    return res

def print_scores(p, p1, r,r1, f1, f11,pn, rn, fn,NO_OF_FOLDS):
    
    print ("None average results are:")
    print (mult(prod(pn, NO_OF_FOLDS),100))
    print (mult(prod(rn, NO_OF_FOLDS),100))
    print (mult(prod(fn, NO_OF_FOLDS),100))       
    
    print ("weighted results are")
    print ("average precision is" , p/NO_OF_FOLDS*100)
    print ("average recall is" , r/NO_OF_FOLDS*100)
    print ("average f1 is" , f1/NO_OF_FOLDS*100)

    print ("macro results are")
    print ("average precision is" , p1/NO_OF_FOLDS*100)
    print ("average recall is" , r1/NO_OF_FOLDS*100)
    print ("average f1 is" , f11/NO_OF_FOLDS*100)
     
        
def create_model( wordEmb, vocab):
    word2vec_model1 = wordEmb.reshape((wordEmb.shape[0], wordEmb.shape[1]))
    word2vec_model = {}
    for k,v in vocab.items():
        word2vec_model[k] = word2vec_model1[int(v)]
    del word2vec_model1
    return word2vec_model 

def Holdout_partition(data,train_users): #seleccionando para training los usuarios mas prolíferos
    print('len(users_none)')
    train_index = []
    test_index = []
    clasestrain = {}
    clasestest ={}
    none = 0
    
    for i in range(len(data)):

        if data[i]['name'] in train_users:            
            train_index.append(i)
        else:    
        #elif data[i]['label'] != 'racism':
            test_index.append(i)
  
    return train_index, test_index


def select_tweets_whose_embedding_exists(tweets, word2vec_model):
    X, Y = [], []
    tweet_return = []
    for tweet in tweets:
        _emb = 0
        words = glove_tokenize(tweet['text'])
        for w in words:
            if w in word2vec_model:  # Check if embeeding there in GLove model
                _emb+=1
        if _emb:   # Not a blank tweet
            tweet_return.append(tweet)
    print ('Tweets selected:', len(tweet_return))
#     print(len(tweet_return))
    return tweet_return 


# def get_embedding_weights(vocab):
#     embedding = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
#     n = 0
#     for k, v in vocab.items():
#         try:
#             embedding[v] = ita_embeddings[ita_word2id[k]]
#         except:
#             try:
#                 embedding[v] = es_embeddings[es_word2id[k]]
#                 print('ok')
#             except:
#                 try:
#                     embedding[v] = en_embeddings[en_word2id[k]]
#                     print('ok')
#                 except:
#                     n += 1
#                     pass
#     print ("%d embedding missed"%n)
#     return embedding
def save_model(model_name,model):
    model_json = model.to_json()
    with open("model_name.json", "wb") as json_file:
        json_file.write(model_json)
    #serializan los pesos (weights) para HDF5
    model.save_weights("model.h5")
    print("Modelo guardado en el PC")
    
def load_model(model_name):
    # carga el json y crea el modelo
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # se cargan los pesos (weights) en el nuevo modelo
    loaded_model.load_weights("model.h5")
    return loaded_model

def save_pickle(pkl_filename,model):
    # Save to file in the current working directory
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
    print('Modelo salvado')

def load_pickle(pkl_filename):
    # Load from file
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model

def get_embedding_weights2(vocab):
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('..\Vectors\glove.twitter.27B.200d.txt')
    embedding = np.zeros((len(vocab) + 1, 200))
    n = 0
    for k, v in vocab.items():
        try:
            embedding[v] = word2vec_model[k]
        except:
            n += 1
            pass
#     print "%d embedding missed"%n
    return embedding


def get_embedding_weights(vocab,vector_type,embed_size):
#     embed = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
#     file = open('..\Vectors\glove.twitter.27B.200d.txt','r', encoding="utf8")
#     for line in file.readlines():
#         row = line.strip().split('\t')
#         embed[row[0]] = row[1:]
#    # print('Loaded from file: ' + str(filename))
#     file.close()
    
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('..\Vectors\glove.twitter.27B.200d.txt')
    embed = np.zeros((len(vocab) , EMBEDDING_DIM))
    embed_size = 300
    vocab_size = len(vocab)
    embeddingWeights = np.zeros((vocab_size , 300))
    n = 0
    words_missed = []
    vector_en = np.zeros(embed_size)
    vector_es = np.zeros(embed_size)
    vector_ita = np.zeros(embed_size)
    print('Len(vocab): ',len(vocab))
    palabras_repeted = []
#     print(len(vocab.items()))
    for k, v in vocab.items():
        found = 0
        if vector_type == 'multilingual':
            try:
                vector_en = en_embeddings[en_word2id[k]]
                found = found + 1
            except:
                pass
            try:
                vector_es = es_embeddings[es_word2id[k]]
                found = found + 1
                if found > 1:
                    palabras_repeted.append(es_word2id[k])
            except:
                pass
            try:
                vector_ita = ita_embeddings[ita_word2id[k]]
                found = found + 1
                if found > 1:
                    palabras_repeted.append(ita_word2id[k])
            except:
                n += 1
                words_missed.append(k)
                pass
            vector = vector_ita + vector_en + vector_es
            if found != 0:
                total = 1/found
                palabras_repeted.append(v)
                embeddingWeights[v] = prod(vector, total) 
           # print("%d embedding missed"%n, " of " , vocab_size)
        else:
            n = 0
            try:
                embed[v] = word2vec_model[k]
            except:
                n += 1
                pass
    return embeddingWeights


def gen_vocab2(X_train):
    # Processing
    vocab, reverse_vocab = {}, {}

    vocab_index = 1
    for tweet in X_train:
        text = TOKENIZER(tweet.lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]

        for word in words:
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word       # generate reverse vocab as well
                vocab_index += 1
            freq[word] += 1
    vocab['UNK'] = len(vocab) + 1
    reverse_vocab[len(vocab)] = 'UNK'
    return vocab


def gen_vocab(tweets):
    # Processing
    vocab, reverse_vocab = {}, {}

    vocab_index = 1
    for tweet in tweets:
        text = TOKENIZER(tweet['text'].lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]

        for word in words:
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word       # generate reverse vocab as well
                vocab_index += 1
            freq[word] += 1
    vocab['UNK'] = len(vocab) + 1
    reverse_vocab[len(vocab)] = 'UNK'
    return vocab


def get_labels(tweets):
    # Processing
    labels = []
    for tweet in tweets:
        labels.append(tweet['label'])
    return labels

def select_tweets(dataset,strategy):
    # selects the tweets as in mean_glove_embedding method
    # Processing
    tweets,users_none = get_data_waseem4(dataset, strategy)
    X, Y = [], []
    tweet_return = []
    for tweet in tweets:
        _emb = 0
        words = glove_tokenize(tweet['text'].lower())
        for w in words:
            #if w in word2vec_model:
            if w in ita_word2id.keys() or w in es_word2id.keys() or w in en_word2id.keys():  # Check if embeeding there in GLove model
                _emb+=1
        if _emb:   # Not a blank tweet
            tweet_return.append(tweet)
    #pdb.set_trace()
    return tweet_return,users_none

def prod(list, num):
    res=[]
    for i in list:
        res.append(i/num)
    return res

def build_corpus_avg(tweets,flag):

    n = 0
    words_missed = []
    vector_en = np.zeros(300)
    vector_es = np.zeros(300)
    vector_ita = np.zeros(300)
    palabras_repeted = []
    if flag == 'binary':
        y_map = {
                'none': 0,
                'racism': 1,
                'sexism': 1,
                'hate':1,
                '1':1,
                '2':2,
                '0':0
            
                }
    else:
        y_map = {
                'none': 0,
                'racism': 1,
                'sexism': 2,
                'hate': 1,
                '1':1,
                '2':2,
                '0':0
                }

    X = []
    y = []
    embeddingWeights =  np.zeros((len(tweets),300), dtype=K.floatx())
    for tweet in range(len(tweets)):
        y.append(y_map[tweets[tweet]['label']])
        n=0
        words = glove_tokenize(tweets[tweet]['text'])
        for k in words:
            found = 0
            try:
                vector_en = en_embeddings[en_word2id[k]]
                found = found + 1
            except:
                pass
            try:
                vector_es = es_embeddings[es_word2id[k]]
                found = found + 1
                if found > 1:
                    word = k 
            except:
                pass
            try:
                vector_ita = ita_embeddings[ita_word2id[k]]
                found = found + 1
                if found > 1:
                    word = k 
            except:
                n += 1
                words_missed.append(k)
                pass
            vector = vector_ita + vector_en + vector_es
            if found != 0:
                total = 1/found
                embeddingWeights[tweet,:] = prod(vector,found) 
                palabras_repeted.append(word)
#     print('Palabras repetidas en ambos idiomas: ',palabras_repeted)
    return embeddingWeights, np.array(y)

def gen_data(tweets_list, word2vec_model,flag):
    if flag == 'binary':
        y_map = {
                'none': 0,
                'racism': 1,
                'sexism': 1,
                'hate':1,
                '1':1,
                '2':2,
                '0':0
            
                }
    else:
        y_map = {
                'none': 0,
                'racism': 1,
                'sexism': 2,
                'hate': 1,
                '1':1,
                '2':2,
                '0':0
                }

    X, y = [], []
    word_embed_size = 200
    for tweet in tweets_list:
        words = glove_tokenize(tweet['text'])
        emb = np.zeros(word_embed_size)
        for word in words:
            try:
                emb += word2vec_model[word]
            except:
                pass
        emb /= len(words)
        X.append(emb)
        y.append(y_map[tweet['label']])
    X = np.array(X)
    y = np.array(y)
    return X, y


def max_len(tweets):
    max_len = 0
    for tweet in tweets:
        text = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(tweet['text'].lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        if len(words) > max_len:
            max_len = len(words)
    return max_len


def max_len2(tweets):
    max_len = 0
    for tweet in tweets:
        text = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(tweet.lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        if len(words) > max_len:
            max_len = len(words)
    return max_len

def gen_sequence(tweets,vocab,flag):
    if flag == 'binary':
        y_map = {
                'none': 0,
                'racism': 1,
                'sexism': 1,
                'hate':1,
                '1':1,
                '2':2,
                '0':0

            
                }
    else:
        y_map = {
                'none': 0,
                'racism': 1,
                'sexism': 2,
                'hate':1,
                '1':1,
                '2':2,
                '0':0

                }
    
    X, y = [], []
    for tweet in tweets:
        text = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(tweet['text'].lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        seq, _emb = [], []
        for word in words:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
        y.append(y_map[tweet['label']])
    return X, y


def gen_sequence2(tweets,Y,vocab,flag):
    if flag == 'binary':
        y_map = {
                'none': 0,
                'racism': 1,
                'sexism': 1,
                'hate':1,
                '1':1,
                '2':2,
                '0':0

            
                }
    else:
        y_map = {
                'none': 0,
                'racism': 1,
                'sexism': 2,
                'hate':1,
                '1':1,
                '2':2,
                '0':0

                }
    
    X, y = [], []
    for tweet in tweets:
        text = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(tweet.lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        seq, _emb = [], []
        for word in words:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
    for i in Y:
        y.append(y_map[i])
    return X, y



def cv_sorted_data(x_text):
    train_indexes = []
    part0,part1,part2,part3,part4,part5,part6,part7,part8,part9 =[],[],[],[],[],[],[],[],[],[]
    for i in range(len(x_text)):
        if i >=0 and i <701:
            part0.append(i)
        elif i >=701 and i <1400:
            part1.append(i)

        elif i >=1400 and i <2102:   
            part2.append(i)

        elif i >=2102 and i <2800:
            part3.append(i)

        elif i >=2800 and i <3500:
            part4.append(i)

        elif i >=3500 and i <4200:
            part5.append(i)
            
        elif i >=4200 and i <4798:
            part6.append(i)
            
        elif i >=4798 and i <5597:
            part7.append(i)
            
        elif i >=5597 and i <6301:
            part8.append(i)
            
        elif i >=6301 and i <7006:
            part9.append(i)
    train_indexes.append(part0) 
    train_indexes.append(part1) 
    train_indexes.append(part2) 
    train_indexes.append(part3) 
    train_indexes.append(part4) 
    train_indexes.append(part5) 
    train_indexes.append(part6) 
    train_indexes.append(part7) 
    train_indexes.append(part8) 
    train_indexes.append(part9) 
    return train_indexes  

def load_data(dataset):
    if flag == 'binary':
        y_map = {
                'none': 0,
                'racism': 1,
                'sexism': 1,
                'hate':1,
                '1':1,
                '2':2,
                '0':0
            
                }
    else:
        y_map = {
                'none': 0,
                'racism': 1,
                'sexism': 2,
                'hate': 1,
                '1':1,
                '2':2,
                '0':0
                }

    
    
    x_text = []
    labels = []
    if dataset =='waseem':
#         print("Loading data from file: " + dataset)
        data = pickle.load(open('../Data/Waseem_Dataset.pkl', 'rb'))
    
    elif dataset == 'ita':
        data = pickle.load(open('../Data/italiano.pkl', 'rb'))
    
    elif dataset == 'portu':
        data = pickle.load(open('../Data/portugues.pkl', 'rb'))
    
    elif dataset == 'italiano_ingles':
        data = pickle.load(open('../Data/italiano_ingles.pkl', 'rb'))
        
    elif dataset == 'espanol_ingles':
        data = pickle.load(open('../Data/espanol_ingles.pkl', 'rb'))
    
    elif dataset == 'esp':
        data = pickle.load(open('../Data/espanol.pkl', 'rb'))
    
    elif dataset == 'sem_eval':
        print("Loading data from file: " + dataset)
#         with open('../Data/train_en.tsv', 'r', encoding = 'utf-8') as sem_file:
#             data = sem_file.readlines()
        data = pickle.load(open('../Data/SemEval_Dataset.pkl', 'rb'))
   
    elif dataset == 'espanol_translated_train':
        data = pickle.load(open('../Data/espanol_translated_train.pkl', 'rb'))
        
    elif dataset == 'espanol_translated_test':
        data = pickle.load(open('../Data/espanol_translated_test.pkl', 'rb'))
    
    elif dataset == 'italiano_translated_train':
        data = pickle.load(open('../Data/italiano_translated_train.pkl', 'rb'))
    
    elif dataset == 'italiano_translated_test':
        data = pickle.load(open('../Data/italiano_translated_test.pkl', 'rb'))
        
            
    elif dataset == 'espanol_train':
        data = pickle.load(open('../Data/tweets_train_esp.pkl', 'rb'))
        
    elif dataset == 'espanol_test':
        data = pickle.load(open('../Data/tweets_test_esp.pkl', 'rb'))
    
    elif dataset == 'italiano_train':
        data = pickle.load(open('../Data/tweets_train_ita.pkl', 'rb'))
    
    elif dataset == 'italiano_test':
        data = pickle.load(open('../Data/tweets_test_ita.pkl', 'rb'))        
        
        
    elif dataset == 'data_new':
        print("Loading data from file: " + dataset)
        data = pickle.load(open('../Data/Data_new.pkl', 'rb')) 
 
    for i in range(len(data)):
        x_text.append(data[i]['text'])
        labels.append(y_map(data[i]['label']))

#     from collections import Counter
#     print(Counter(labels))
    return x_text, labels

def get_data_waseem4(dataset, s):
    tweets=[]
    if dataset == 'data_new':
        data = pickle.load(open('../Data/Data_new.pkl', 'rb'))
    elif dataset == 'waseem':
        data = pickle.load(open('../Data/Waseem_Dataset.pkl', 'rb'))
    elif dataset == 'esp':
        data = pickle.load(open('../Data/espanol.pkl', 'rb'))
    elif dataset == 'espanol_ingles':
        data = pickle.load(open('../Data/espanol_ingles.pkl', 'rb'))
    elif dataset == 'italiano_ingles':
        data = pickle.load(open('../Data/italiano_ingles.pkl', 'rb'))
    elif dataset == 'portu':
        data = pickle.load(open('../Data/portugues.pkl', 'rb'))
    elif dataset == 'ita':
        data = pickle.load(open('../Data/italiano.pkl', 'rb'))
    
    elif dataset == 'espanol_translated_train':
        data = pickle.load(open('../Data/espanol_translated_train.pkl', 'rb'))
        
    elif dataset == 'espanol_translated_test':
        data = pickle.load(open('../Data/espanol_translated_test.pkl', 'rb'))
    
    elif dataset == 'italiano_translated_train':
        data = pickle.load(open('../Data/italiano_translated_train.pkl', 'rb'))
    
    elif dataset == 'italiano_translated_test':
        data = pickle.load(open('../Data/italiano_translated_test.pkl', 'rb'))
     
    elif dataset == 'espanol_train': 
        data = pickle.load(open('../Data/tweets_train_esp.pkl', 'rb'))
        
    elif dataset == 'espanol_test':
        data = pickle.load(open('../Data/tweets_test_esp.pkl', 'rb'))
    
    elif dataset == 'italiano_train':
        data = pickle.load(open('../Data/tweets_train_ita.pkl', 'rb'))
    
    elif dataset == 'italiano_test':
        data = pickle.load(open('../Data/tweets_test_ita.pkl', 'rb'))    
    
    elif dataset == 'sem_eval':
        data = pickle.load(open('../Data/SemEval_Dataset.pkl', 'rb'))
        
        with open('../Data/train_en.tsv', 'r', encoding='utf-8') as f_in:
            data = f_in.readlines()
            for tweet_full in data[1:len(data)]:
                tweet = tweet_full.split('\t')
                if tweet[2] == '0':
                    label = 'none'
                else:
                    label = 'hate'
                tweets.append({
                    'id': tweet[0],#id en el dataset, porque el de el tweet no se conoce
                    'text': tweet[1].lower(),
                    'label': label
                    })
            return tweets, []

    
    Odiosos=[]
    none=0
    dict_users_none={}
    dict_users_sexist={}
    dict_users_racist={}
    strategy = s
    for tweet_full in data:
        #tweet_full = json.loads(line)
        tweets.append({
            #'id': tweet_full['id'],
            'text': tweet_full['text'].lower(),
            'label': tweet_full['label'],
            'name': tweet_full['name']
            })
        if tweet_full['label'] != 'none':
            Odiosos.append(tweet_full['name'])
        if tweet_full['label'] == 'none':
            none+=1
            if tweet_full['name'] in dict_users_none.keys():
                dict_users_none[tweet_full['name']] += 1
            else:
                dict_users_none[tweet_full['name']] = 1 
                
        if tweet_full['label'] == 'sexism':
            none+=1
            if tweet_full['name'] in dict_users_sexist.keys():
                dict_users_sexist[tweet_full['name']] += 1
            else:
                dict_users_sexist[tweet_full['name']] = 1 
                
        if tweet_full['label'] == 'racism':
            none+=1
            if tweet_full['name'] in dict_users_racist.keys():
                dict_users_racist[tweet_full['name']] += 1
            else:
                dict_users_racist[tweet_full['name']] = 1 
                
    resultado = sorted(dict_users_none.items(), key=operator.itemgetter(1))
    resultado.reverse()
    
    users_train = []
    for i in resultado[:1400]:
        if i[0] not in Odiosos:
            users_train.append(i[0])


    None_users = sorted(dict_users_none.items(), key=operator.itemgetter(1))
    None_users.reverse()
    Sexist_users = sorted(dict_users_sexist.items(), key=operator.itemgetter(1))
    Sexist_users.reverse()
    Racist_users = sorted(dict_users_racist.items(), key=operator.itemgetter(1))
    Racist_users.reverse()   
    print('cantidad de usuarios en dataset')
    print(len(Racist_users) + len(Sexist_users) + len(None_users))
    
    if strategy == 1:
        print('strategy')
        print(strategy)
        t = [Sexist_users[0][0],Racist_users[0][0],Sexist_users[1][0]]
        for i in t:
            users_train.append(i)
            
        resultado = sorted(dict_users_sexist.items(), key=operator.itemgetter(1))
        resultado.reverse()


        resultado = sorted(dict_users_racist.items(), key=operator.itemgetter(1))
        resultado.reverse()


    elif strategy == 2:
        print('strategy')
        print(strategy)
        for i in resultado:
            if i[0] not in Odiosos:
                users_train.append(i[0])

        t = [Racist_users[0][0]]
        for i in t:
            users_train.append(i)
            
        resultado = sorted(dict_users_none.items(), key=operator.itemgetter(0))
        resultado.reverse()
    
        count =0
        resultado = sorted(dict_users_sexist.items(), key=operator.itemgetter(0))
        resultado.reverse() 
        for i in resultado:
            if i[0] != Sexist_users[1][0] and i[0] not in  dict_users_racist.keys() and (count < 160):
                users_train.append(i[0]) 
            count += 1
        for i in sorted(dict_users_sexist.items(), key=operator.itemgetter(1)):
            if i[0] not in dict_users_none.keys():
                users_train.append(i) 
                count += 1
    
    elif strategy == 3:
        print('strategy')
        print(strategy)

        for i in resultado:
             if i[0] not in Odiosos:
                users_train.append(i[0])

        t = [Racist_users[0][0]]
        for i in t:
            users_train.append(i)

        count =0
        resultado = sorted(dict_users_sexist.items(), key=operator.itemgetter(0))
        resultado.reverse() 
        for i in resultado:
            if i[0] != Sexist_users[0][0] and i[0] not in  dict_users_racist.keys():
                users_train.append(i[0]) 
            count += 1
    
    return tweets,users_train
