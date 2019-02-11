import argparse
import keras
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, LSTM
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Convolution1D, MaxPooling1D, GlobalMaxPooling1D
import numpy as np
import pdb
import pandas as pd
import pickle
from tensorflow.contrib import learn
MAX_FEATURES = 2
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.model_selection import KFold
from models import get_model
import string
#from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
import preprocessor as p
from collections import Counter
import os
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix 
from scipy import stats
import tflearn
from nltk import tokenize as tokenize_nltk
TOKENIZER = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
import operator
import random
from gensim.parsing.preprocessing import STOPWORDS
from keras.utils import np_utils
import codecs
from string import punctuation
from collections import defaultdict
import sys
from models import get_model

models = ['cnn', 'lstm', 'blstm', 'blstm_attention']
word_vectors = ["random", "glove" ,"sswe"]
EPOCHS = 10
BATCH_SIZE = 512
MAX_FEATURES = 2
NUM_CLASSES = None
DROPOUT = 0.25
LEARN_RATE = 0.01
HASH_REMOVE = None
output_folder_name = "results/"


def prod(list, num):
    res=[]
    for i in list:
        res.append(i/num)
    return res

def get_embeddings_dict(vector_type, emb_dim):
    if vector_type == 'sswe':
        emb_dim==50
        sep = '\t'
        vector_file = 'word_vectors/sswe-u.txt'
    elif vector_type =="glove":
        sep = ' '
        if data == "wiki":
            vector_file = 'word_vectors/glove.6B.' + str(emb_dim) + 'd.txt'
        else:
            vector_file = 'word_vectors/glove.twitter.27B.' + str(emb_dim) + 'd.txt'
    else:
        print ("ERROR: Please specify a correst model or SSWE cannot be loaded with embed size of: " + str(emb_dim) )
        return None
    
    embed = get_embedding_weights(vector_file, sep)
    return embed

def map_embedding_weights(embed, vocab, embed_size):
    vocab_size = len(vocab)
    embeddingWeights = np.zeros((vocab_size , embed_size))
    n = 0
    words_missed = []
    for k, v in vocab.items():
        try:
            embeddingWeights[v] = embed[k]
        except:
            n += 1
            words_missed.append(k)
            pass
    print("%d embedding missed"%n, " of " , vocab_size)
    return embeddingWeights

def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)

def oversampling(x_text,labels,oversampling_rate):
    NUM_CLASSES = 3
    racism = [i for i in range(len(labels)) if labels[i]==2]
    sexism = [i for i in range(len(labels)) if labels[i]==1]
    print('len(racism)')
    print(len(racism))
    print('len(sexism)')
    print(len(sexism))
    x_text = x_text + [x_text[x] for x in racism]*(oversampling_rate-1)+ [x_text[x] for x in sexism]*(oversampling_rate-1)
    labels = labels + [2 for i in range(len(racism))]*(oversampling_rate-1) + [1 for i in range(len(sexism))]*(oversampling_rate-1)

    print("Counter afterrrrr oversampling")
    from collections import Counter
    print(Counter(labels))
    return x_text, labels

def get_embedding_weights(filename, sep):
    embed_dict = {}
    file = open(filename,'r', encoding="utf8")
    for line in file.readlines():
        row = line.strip().split(sep)
        embed_dict[row[0]] = row[1:]
    print('Loaded from file: ' + str(filename))
    file.close()
    return embed_dict

def evaluate_model(model, testX, testY,flag):
    print("evaluate_model")   
    print(flag)
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
        y_true = np.argmax(testY, axis=1)
        y_pred = model.predict(testX)
        y_pred = np.argmax(y_pred, axis=1)
    
    elif flag == 'cross_domain_waseem':
        temp = model.predict(testX)
        temp = np.argmax(temp, axis=1)
        y_true = np.argmax(testY, axis=1)
        y_pred=[]
        for i in temp:
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
    print(confusion_matrix(y_true, y_pred))
    print(":: Classification Report")
    print(classification_report(y_true, y_pred))
    return precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem

def return_data(data_dict):
    return data_dict["data"], data_dict["trainX"], data_dict["trainY"], data_dict["testX"], data_dict["testY"],data_dict["vocab_processor"]

def train(data_dict, model_type, vector_type,flag, embed_size, dump_embeddings=False):
    print("trainnn")   
    print(flag)
    data, trainX, trainY, testX, testY, vocab_processor = return_data(data_dict)
    
    vocab_size = len(vocab_processor.vocabulary_)
    NUM_CLASSES = 3
    print(NUM_CLASSES)
    print(trainX.shape[1])
    print("Vocabulary Size: {:d}".format(vocab_size))
    vocab = vocab_processor.vocabulary_._mapping

    print("Running Model: " + str(model_type) + " with word vector initiliazed with " + str(vector_type) + " word vectors.")
    model = get_model(model_type, trainX.shape[1], vocab_size, embed_size, NUM_CLASSES, LEARN_RATE)

    initial_weights = model.get_weights()
    shuffle_weights(model, initial_weights)
    
    if(model_type == 'cnn'):
        if(vector_type!="random"):
            print("Word vectors used: " + vector_type)
            embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
            model.set_weights(embeddingWeights, map_embedding_weights(get_embeddings_dict(vector_type, embed_size), vocab, embed_size))
            model.fit(trainX, trainY, n_epoch = EPOCHS, shuffle=True, show_metric=True, batch_size=BATCH_SIZE)
        else:
            model.fit(trainX, trainY, n_epoch = EPOCHS, shuffle=True, show_metric=True, batch_size=BATCH_SIZE)
    else:
        if(vector_type!="random"):
            print("Word vectors used: " + str(vector_type))
            model.layers[0].set_weights([map_embedding_weights(get_embeddings_dict(vector_type, embed_size), vocab, embed_size)])
            model.fit(trainX, trainY, epochs=EPOCHS, shuffle=True, batch_size=BATCH_SIZE, 
                  verbose=1)
        else:
            model.fit(trainX, trainY, epochs=EPOCHS, shuffle=True, batch_size=BATCH_SIZE, 
                  verbose=1)
            
    if (dump_embeddings==True):
        if(model_type == 'cnn'):
            embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
        else:
            embed = model.layers[0].get_weights()[0]
    
        embed_filename = output_folder_name + data + "_" + model_type + "_" + vector_type + "_" + str(embed_size) + ".pkl"
        embed.dump(embed_filename)
        
        vocab_filename = output_folder_name + data + "_" + model_type + "_" + vector_type + "_" + str(embed_size) + "_dict.json"
        reverse_vocab_filename = output_folder_name + data + "_" + model_type + "_" + vector_type + "_" + str(embed_size) + "_reversedict.json"
        
        with open(vocab_filename, 'w') as fp:
            json.dump(vocab_processor.vocabulary_._mapping, fp)
        with open(reverse_vocab_filename, 'w') as fp:
            json.dump(vocab_processor.vocabulary_._reverse_mapping, fp)
    return  evaluate_model(model, testX, testY,flag)

def print_scores(p, p1, r,r1, f1, f11,pn, rn, fn,NO_OF_FOLDS):
    print ("None results are")
    print (prod(pn, NO_OF_FOLDS))
    print (prod(rn, NO_OF_FOLDS))
    print (prod(fn, NO_OF_FOLDS))    
        
    print ("weigthed results are")
    print ("average precision is %f" %(p/NO_OF_FOLDS))
    print ("average recall is %f" %(r/NO_OF_FOLDS))
    print ("average f1 is %f" %(f1/NO_OF_FOLDS))

    print ("micro results are")
    print ("average precision is %f" %(p1/NO_OF_FOLDS))
    print ("average recall is %f" %(r1/NO_OF_FOLDS))
    print ("average f1 is %f" %(f11/NO_OF_FOLDS))

def get_filename(dataset):
    global NUM_CLASSES, HASH_REMOVE
    if(dataset=="twitter"):
        NUM_CLASSES = 2
        HASH_REMOVE = True
        filename = "DatosCSV/twitter_data.pkl"
    elif(dataset=="formspring"):
        NUM_CLASSES = 2
        filename = "data/formspring_data.pkl"
    elif(dataset=="wiki"):
        NUM_CLASSES = 2
        filename = "data/wiki_data.pkl"
    return filename


def load_data(dataset):
    if dataset =='train':
        data = pickle.load(open('DatosCSV/waseem4.pkl', 'rb'))
    elif dataset == 'test':
        #data = pickle.load(open('DatosCSV/twitter_data_davidson.pkl', 'rb'))    
        data = pickle.load(open('DatosCSV/sem_eval.pkl', 'rb'))   
    elif dataset == 'data_new':
        data = pickle.load(open('DatosCSV/data_new.pkl', 'rb')) 
        print(len(data))
    elif dataset == 'data_sorted':
        data = pickle.load(open('DatosCSV/data_sorted.pkl', 'rb')) 
        print(len(data))
    x_text = []
    labels = [] 
    for i in range(len(data)):
        if(HASH_REMOVE):
            x_text.append(data[i]['text'])
        else:
            x_text.append(data[i]['text'])
        labels.append(data[i]['label'])
        
        
    filter_data = []
    i=0
    for text in x_text:
        try:
            filter_data.append("".join(l for l in text if l not in string.punctuation))
        except Exception as e:
            print(i)
            print(text)
        i += 1
    dict1 = {'racism':1,'sexism':2,'none':0,'hate':1}
    labels = [dict1[b] for b in labels]
    return x_text,labels

def data_processor(x_text,X_train,y_train,X_test,y_test,flag):
    post_length = np.array([len(x.split(" ")) for x in x_text])
    max_document_length = int(np.percentile(post_length, 95))
    print("Document length : " + str(max_document_length))

    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, MAX_FEATURES)
    vocab_processor = vocab_processor.fit(x_text) 
    trainX = np.array(list(vocab_processor.transform(X_train)))
    testX = np.array(list(vocab_processor.transform(X_test)))

    trainY = np.asarray(y_train)
    testY = np.asarray(y_test)

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    
    if flag != 'binary':
        trainY = to_categorical(trainY, nb_classes=3)
        testY = to_categorical(testY, nb_classes=3)
        
    data_dict = {
    "data": "twitter",
    "trainX" : trainX,
    "trainY" : trainY,
    "testX" : testX,
    "testY" : testY,
    "vocab_processor" : vocab_processor
    }
    return data_dict


def gen_sequence(tweets):
    y_map = {
            'none': 0,
            'racism': 1,
            'sexism': 2,
            'hate' :1
            }

    X, y = [], []
    for tweet in tweets:
        text = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(tweet['text'].lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        #print(len(words))
        seq, _emb = [], []
        for word in words:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
        y.append(y_map[tweet['label']])
    print('len(vocab)')
    print(len(vocab))

    return X, y

def cv_sorted_data(x_text):
    train_indexes = []
    part0,part1,part2,part3,part4,part5,part6,part7,part8,part9 =[],[],[],[],[],[],[],[],[],[]
    for i in range(len(x_text)):
        if i >=0 and i <700:
            part0.append(i)
        elif i >=700 and i <1400:
            part1.append(i)

        elif i >=1400 and i <2100:   
            part2.append(i)

        elif i >=2100 and i <2800:
            part3.append(i)

        elif i >=2800 and i <3498:
            part4.append(i)

        elif i >=3498 and i <4200:
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
    
def get_data_waseem3(s):
    data = pickle.load(open('DatosCSV/waseem3.pkl', 'rb'))
    tweets=[]
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
            if tweet_full['name'] in dict_users_none.keys():
                dict_users_none[tweet_full['name']] += 1
            else:
                dict_users_none[tweet_full['name']] = 1 
                
        if tweet_full['label'] == 'sexism':
            if tweet_full['name'] in dict_users_sexist.keys():
                dict_users_sexist[tweet_full['name']] += 1
            else:
                dict_users_sexist[tweet_full['name']] = 1 
                
        if tweet_full['label'] == 'racism':
            if tweet_full['name'] in dict_users_racist.keys():
                dict_users_racist[tweet_full['name']] += 1
            else:
                dict_users_racist[tweet_full['name']] = 1 
                
    resultado = sorted(dict_users_none.items(), key=operator.itemgetter(1))
    resultado.reverse()


    if strategy == 1:
        users_train = []
        print('strategy')
        print(strategy)
        for i in resultado[:1400]:
            if i[0] not in Odiosos:
                users_train.append(i[0])

        t = ["Yes You're Sexist","Levi Stein","needlessly obscenity-laced","Male Tears #4648"]
        for i in t:
            users_train.append(i)
        print('len(users_train)')
        print(len(users_train))
    elif strategy == 2:
        print('strategy')
        print(strategy)
        users_train = []
        for i in resultado:
            if i[0] not in Odiosos:
                users_train.append(i[0])
        users_train = []
        users_train.append("Levi Stein")
        resultado = sorted(dict_users_sexist.items(), key=operator.itemgetter(0))
        resultado.reverse() 
        for i in resultado[0:160]:
            if i[0] != "Male Tears #4648" and i[0] not in  dict_users_racist.keys():
                users_train.append(i[0]) 
        print('len(users_train)')
        print(len(users_train))    
    elif strategy == 3:
        users_train = []
        print('strategy')
        print(strategy)
        for i in resultado:
             if i[0] not in Odiosos:
                users_train.append(i[0])

        t = ["Levi Stein","needlessly obscenity-laced"]
        for i in t:
            users_train.append(i)
            
        resultado = sorted(dict_users_none.items(), key=operator.itemgetter(0))
        resultado.reverse()
    
        count =0
        resultado = sorted(dict_users_sexist.items(), key=operator.itemgetter(0))
        resultado.reverse() 
        for i in resultado:
            if i[0] != "Yes You're Sexist" and i[0] not in  dict_users_racist.keys():
                users_train.append(i[0]) 
            count += 1
    elif strategy == 4: #ignorando clase racista
        resultado = sorted(dict_users_none.items(), key=operator.itemgetter(1))
        resultado.reverse()
        print('strategy')
        print(strategy)
        print(len(dict_users_racist.keys() ))
        print(len(dict_users_sexist.keys() ))
        print(len(dict_users_none.keys() ))
        users_train = []
        for i in resultado[:1400]:
            if i[0] not in dict_users_racist.keys() and i[0] not in dict_users_sexist.keys():
                users_train.append(i[0])
        t = ["needlessly obscenity-laced"]
        for i in t:
            users_train.append(i)

        resultado = sorted(dict_users_sexist.items(), key=operator.itemgetter(0))
        resultado.reverse() 
        for i in resultado:
            if i[0] != "Yes You're Sexist" and i[0] not in dict_users_racist.keys():
                users_train.append(i[0]) 
                
    return tweets,users_train
    
    
def Holdout_partition(oversampling_rate,strategy,flag): #seleccionando para training los usuarios mas prolíferos
    data = pickle.load(open('DatosCSV/waseem3.pkl', 'rb'))
    _,train_users = get_data_waseem3(strategy)
    
    train_index = []
    x_text=[]
    labels=[]
    x_text_train = []
    labels_train = [] 
    x_text_test = []
    labels_test = [] 
    test_index = []
    clasestrain = {}
    clasestest ={}
    none = 0
    #train_users = ["Yes You're Sexist","Levi Stein","needlessly obscenity-laced","Male Tears #4648"]
    for i in range(len(data)):
        if data[i]['name'] in train_users:        
            train_index.append(i)
        else:
            test_index.append(i)
    
    for i in range(len(data)):
        if i in train_index:
            x_text_train.append(data[i]['text'])
            labels_train.append(data[i]['label'])
        else:
            x_text_test.append(data[i]['text'])
            labels_test.append(data[i]['label'])
            
            
    if flag == 'binary':
        dict1 = {'sexism':1,'racism':1,'none':0}
    else:
        dict1 = {'sexism':2,'racism':1,'none':0}
        
    #NUM_CLASSES = 1
    labels_train = [dict1[b] for b in labels_train]
    print("Counter TRAIN")
    from collections import Counter
    print(Counter(labels_train))
    
    labels_test = [dict1[b] for b in labels_test]
    print("Counter test")
    from collections import Counter
    print(Counter(labels_test))
    
    
    x_text_train,labels_train = oversampling(x_text_train,labels_train,oversampling_rate)

    print("Counter TRAIN afterrr")
    from collections import Counter
    print(Counter(labels_train))
          
    x_text = np.concatenate((x_text_train,x_text_test), axis = 0)    
    data_dict = data_processor(x_text,x_text_train,labels_train,x_text_test,labels_test,flag)
    
    return data_dict
