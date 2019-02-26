from data_handler import *
import argparse
import keras
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, LSTM
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Convolution1D, MaxPooling1D, GlobalMaxPooling1D
import numpy as np
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
from batch_gen import batch_gen
import sys
import xgboost as xgb
import json
from nltk import tokenize as tokenize_nltk
from my_tokenizer import glove_tokenize

GLOVE_MODEL_FILE = 'glove.txt'
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_MODEL_FILE)
freq = defaultdict(int)
EMBEDDING_DIM = 200
NO_OF_FOLDS = 10
LEARN_EMBEDDINGS = True
INITIALIZE_WEIGHTS_WITH = 'random'
SCALE_LOSS_FUN = None
TOKENIZER = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize

def gradient_boosting_classifier(tweets, wordEmb, train_index, test_index, X_train, y_train, X_test, y_test,flag):
    a, p, r, f1 = 0., 0., 0., 0.
    a1, p1, r1, f11 = 0., 0., 0., 0.
    pn, rn, fn = 0., 0., 0.
    print('gradient_boosting_classifier')
    X_train = np.array(X_train)
    clf = xgb.XGBClassifier(nthread=-1,n_estimators=200)  
    print('clasificando...')
    clf.fit(X_train, y_train)
    temp = clf.predict(X_test)
    
    if flag =='binary':
        y_pred = temp

    elif flag == 'categorical':
        y_pred = temp

    else:
        y_pred=[]
        for i in temp:
            if i == 2:
                y_pred.append(1)
            else:
                y_pred.append(i)

    print (classification_report(y_test, y_pred))
    print (precision_recall_fscore_support(y_test, y_pred))
    
    a = accuracy_score (y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1_s = f1_score(y_test, y_pred, average=None)
    p = precision_score(y_test, y_pred, average='weighted')
    p1 = precision_score(y_test, y_pred, average='macro')
    r = recall_score(y_test, y_pred, average='weighted')
    r1 = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='weighted')
    f11 = f1_score(y_test, y_pred, average='macro')

    return precision,recall,f1_s,a, p, p1, r, r1, f1, f11


def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)


def print_scores(p, p1, r,r1, f1, f11,pn, rn, fn,NO_OF_FOLDS):
    print ("None average results are:")
    print (prod(pn, NO_OF_FOLDS))
    print (prod(rn, NO_OF_FOLDS))
    print (prod(fn, NO_OF_FOLDS))       
    
    print ("weighted results are")
    print ("average precision is %f" %(p/NO_OF_FOLDS))
    print ("average recall is %f" %(r/NO_OF_FOLDS))
    print ("average f1 is %f" %(f1/NO_OF_FOLDS))

    print ("macro results are")
    print ("average precision is %f" %(p1/NO_OF_FOLDS))
    print ("average recall is %f" %(r1/NO_OF_FOLDS))
    print ("average f1 is %f" %(f11/NO_OF_FOLDS))  
     
    
def select_tweets_whose_embedding_exists(flag, wordEmb):
    # selects the tweets as in mean_glove_embedding method
    # Processing
    word2vec_model1 = wordEmb.reshape((wordEmb.shape[0], wordEmb.shape[1]))
    word2vec_model = {}
    for k,v in vocab.items():
        word2vec_model[k] = word2vec_model1[int(v)]
    del word2vec_model1
    if flag == 'waseem':
        tweets,_ = get_data_waseem3('waseem',1)
    else:
        tweets,_ = get_data_waseem3('sem_eval',1)
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
    #pdb.set_trace()
    return tweet_return, word2vec_model

def get_embedding_weights():
    embedding = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
    n = 0
    for k, v in vocab.items():
        try:
            embedding[v] = word2vec_model[k]
        except:
            n += 1
            pass
    print ("%d embedding missed"%n)

def gen_vocab(tweets):
    # Processing
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

def select_tweets(dataset,strategy):
    # selects the tweets as in mean_glove_embedding method
    # Processing
    tweets,users_none = get_data_waseem3(dataset,strategy)
    X, Y = [], []
    tweet_return = []
    for tweet in tweets:
        _emb = 0
        words = glove_tokenize(tweet['text'].lower())
        for w in words:
            if w in word2vec_model:  # Check if embeeding there in GLove model
                _emb+=1
        if _emb:   # Not a blank tweet
            tweet_return.append(tweet)
    print ('Tweets selected:', len(tweet_return))
    #pdb.set_trace()
    return tweet_return,users_none

def lstm_model_bin(sequence_length, embedding_dim):
    model_variation = 'LSTM'
    print('Model variation is %s' % model_variation)
    model = Sequential()
    print('variables')
    print(embedding_dim)
    print(sequence_length)
    model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length, trainable=LEARN_EMBEDDINGS))
    model.add(Dropout(0.25))#, input_shape=(sequence_length, embedding_dim)))
    model.add(LSTM(embedding_dim))#, input_shape=(sequence_length, embedding_dim)))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print (model.summary())
    return model    
    
def lstm_model(sequence_length, embedding_dim):
    model_variation = 'LSTM'
    print('Model variation is %s' % model_variation)
    model = Sequential()
    print('variables')
    print(embedding_dim)
    print(sequence_length)
    model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length, trainable=LEARN_EMBEDDINGS))
    model.add(Dropout(0.25))#, input_shape=(sequence_length, embedding_dim)))
    model.add(LSTM(embedding_dim))#, input_shape=(sequence_length, embedding_dim)))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print (model.summary())
    return model

def prod(list, num):
    res=[]
    for i in list:
        res.append(i/num)
    return res

def gen_data(tweets_list, word2vec_model,flag):
    if flag == 'binary':
        y_map = {
                'none': 0,
                'racism': 1,
                'sexism': 1,
                'hate':1
                }
    else:
        y_map = {
                'none': 0,
                'racism': 1,
                'sexism': 2,
                'hate': 1
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


def gen_sequence(tweets,vocab,flag):
    if flag == 'binary':
        y_map = {
                'none': 0,
                'racism': 1,
                'sexism': 1,
                'hate': 1,
                }
    else:
        y_map = {
                'none': 0,
                'racism': 1,
                'sexism': 2,
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

def cv_sorted_data(x_text):
    train_indexes = []
    part0,part1,part2,part3,part4,part5,part6,part7,part8,part9 =[],[],[],[],[],[],[],[],[],[]
    for i in range(len(x_text)):
        if i >=0 and i <700:
            part0.append(i)
        elif i >700 and i <1400:
            part1.append(i)

        elif i >=1400 and i <2100:   
            part2.append(i)

        elif i >=2100 and i <2800:
            part3.append(i)

        elif i >=2800 and i <3500:
            part4.append(i)

        elif i >=3500 and i <4200:
            part5.append(i)
        elif i >=4200 and i <4900:
            part6.append(i)
        elif i >=4900 and i <5600:
            part7.append(i)
            
        elif i >=5600 and i <6300:
            part8.append(i)
            
        elif i >=6300 and i <7006:
            part9.append(i)
    train_indexes.append(part9) 
    train_indexes.append(part0) 
    train_indexes.append(part1) 
    train_indexes.append(part2) 
    train_indexes.append(part3) 
    train_indexes.append(part4) 
    train_indexes.append(part5) 
    train_indexes.append(part6) 
    train_indexes.append(part7) 
    train_indexes.append(part8) 
    return train_indexes  

def get_data_waseem3(dataset,s):
    tweets=[]
    print("Loading data from file: " + 'tweet_data/twitter_data.pkl')
    print(dataset)
    if dataset == 'data_new':
        data = pickle.load(open('data_sorted.pkl', 'rb'))
    elif dataset == 'waseem':
        data = pickle.load(open('waseem3.pkl', 'rb'))
    elif dataset == 'sem_eval':
        data = pickle.load(open('tweet_data/data_sorted.pkl', 'rb'))
        for tweet_full in data:
        #tweet_full = json.loads(line)
            tweets.append({
                #'id': tweet_full['id'],
                'text': tweet_full['text'].lower(),
                'label': tweet_full['label'],
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
                
    None_users = sorted(dict_users_none.items(), key=operator.itemgetter(1))
    None_users.reverse()
    Sexist_users = sorted(dict_users_sexist.items(), key=operator.itemgetter(1))
    Sexist_users.reverse()
    Racist_users = sorted(dict_users_racist.items(), key=operator.itemgetter(1))
    Racist_users.reverse()

    users_train = []
    for i in None_users:
        if i[0] not in Odiosos[0:1400]:
            users_train.append(i[0])
    print(users_train[0])
    
    if strategy == 1:
        print('strategy')
        print(strategy)
        print(Racist_users[0][0])
        t = [Sexist_users[0][0],Racist_users[0][0],Sexist_users[1][0]]
        for i in t:
            users_train.append(i)

    elif strategy == 2:
        print('strategy')
        print(strategy)
        print(len(users_train))
        for i in None_users:
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
        for i in None_users:
             if i[0] not in Odiosos:
                users_train.append(i[0])

        t = [Racist_users[0][0]]
        for i in t:
            users_train.append(i)
        count =0
        resultado = sorted(dict_users_sexist.items(), key=operator.itemgetter(0))
        resultado.reverse() 
        for i in resultado:
            if i[0] != "Yes You're Sexist" and i[0] not in  dict_users_racist.keys():
                users_train.append(i[0]) 
            count += 1
            
    return tweets,users_train
