#from data_handler import get_data
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
#from batch_gen import batch_gen
import sys
from auxiliares import *
#rom nltk import tokenize as tokenize_nltk
#from my_tokenizer import glove_tokenize
def Holdout_partition(data, oversampling_rate): #seleccionando para training los usuarios mas prolíferos
    clasestrain = {}
    clasestest ={}
    train_users = ['Levi Stein',"Yes You're Sexist",'Male Tears #4648']#tres usuarios mas odiosos

    print("Loading data from file: " + get_filename(data))
    data = pickle.load(open(get_filename(data), 'rb'))
    x_text=[]
    labels=[]
    x_text_train = []
    labels_train = [] 
    x_text_test = []
    labels_test = [] 
    
    dict1 = {'racism':2,'sexism':1,'none':0}
    for i in data:
        labels.append(i['label'])
        x_text.append(i['text'])       

        
    labels = [dict1[b] for b in labels]

    print("Counter before oversampling ALL")
    from collections import Counter
    print(Counter(labels))
    
    racism = [i for i in range(len(labels)) if labels[i]==2]
    sexism = [i for i in range(len(labels)) if labels[i]==1]
    
    print(len(racism))
    print(len(sexism))
    
    data = data + [data[x] for x in racism]*(oversampling_rate-1)+ [data[x] for x in sexism]*(oversampling_rate-1)
    labels = labels + [2 for i in range(len(racism))]*(oversampling_rate-1) + [1 for i in range(len(sexism))]*(oversampling_rate-1)
     
    print("Counter AFTER oversampling ALL")
    from collections import Counter
    print(Counter(labels))
    
    for i in range(len(data)):
        if data[i]['user'] in train_users:
            x_text_train.append(data[i]['text'])
            labels_train.append(data[i]['label'])
        else:
            x_text_test.append(data[i]['text'])
            labels_test.append(data[i]['label'])
    
    
    NUM_CLASSES = 3
    labels_train = [dict1[b] for b in labels_train]
    print("Counter TRAIN")
    from collections import Counter
    print(Counter(labels_train))
    
    labels_test = [dict1[b] for b in labels_test]
    print("Counter test")
    from collections import Counter
    print(Counter(labels_test))
          
    post_length = np.array([len(x.split(" ")) for x in x_text])
    if(data != "twitter"):
        max_document_length = int(np.percentile(post_length, 95))
    else:
        max_document_length = max(post_length)
    print("Document length : " + str(max_document_length))
    
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, MAX_FEATURES)
    vocab_processor = vocab_processor.fit(x_text)

    trainX = np.array(list(vocab_processor.transform(x_text_train)))
    testX = np.array(list(vocab_processor.transform(x_text_test)))
    
    trainY = np.asarray(labels_train)
    testY = np.asarray(labels_test)
        
    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)

    trainY = to_categorical(trainY, nb_classes=NUM_CLASSES)
    testY = to_categorical(testY, nb_classes=NUM_CLASSES)
    
    data_dict = {
        "data": data,
        "trainX" : trainX,
        "trainY" : trainY,
        "testX" : testX,
        "testY" : testY,
        "vocab_processor" : vocab_processor
    }
    
    return data_dict

def get_data_dict(X_train,X_test,y_train,y_test):
    x_text = np.concatenate((X_train,X_test),axis = 0)
    post_length = np.array([len(x.split(" ")) for x in x_text])
    if(data != "twitter"):
        max_document_length = int(np.percentile(post_length, 95))
    else:
        max_document_length = max(post_length)
    print("Document length : " + str(max_document_length))

    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, MAX_FEATURES)
    vocab_processor = vocab_processor.fit(x_text)

    trainX = np.array(list(vocab_processor.transform(X_train)))
    testX = np.array(list(vocab_processor.transform(X_test)))

    trainY = np.asarray(y_train)
    testY = np.asarray(y_test)

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)

    trainY = to_categorical(trainY, nb_classes=NUM_CLASSES)
    testY = to_categorical(testY, nb_classes=NUM_CLASSES)

    data_dict = {
            "data": data,
            "trainX" : trainX,
            "trainY" : trainY,
            "testX" : testX,
            "testY" : testY,
            "vocab_processor" : vocab_processor
        }
    return data_dict


def user_kfold(data, oversampling_rate): #seleccionando para training los usuarios mas prolíferos
    import pandas as pd
    data = pd.read_csv('DatosCSV\en_total.csv','r',delimiter=',')
    particion1=[]
    X=[]#usuarios
    y=[]#clase
    for line in data.values:
        y.append(line[0]) 
        X.append(line[3])
    X=np.array(X)
    y=np.array(y)

    from sklearn.model_selection import KFold
    KFold(n_splits=3, random_state=0, shuffle=False)
    kf = KFold(n_splits=3)
    kf.get_n_splits(X)
    p=0
    TESTS = []
    print(X[0:4983])
    print(X[4983:9966])
    print(X[9966:len(X)-1])
    print('Chequeando Usuarios Solapados.....')
    for i in X[0:4983]:
        if i in X[4983:9966] or i in X[9966:len(X)-1]:
            print('Usuario Solapadooooooo:' + str(i))
    print('TESTS.shape()')    

    #for train_index, test_index in kf.split(X,y):
    particion1 =[X[0:4983],X[4984:9966],X[9967:len(X)-1]]
    ####################
    
    print("Loading data from file: " + get_filename("twitter"))
    
def get_data_dict(X_train,X_test,y_train,y_test):
    x_text = np.concatenate((X_train,X_test),axis = 0)
    post_length = np.array([len(x.split(" ")) for x in x_text])
    if(data != "twitter"):
        max_document_length = int(np.percentile(post_length, 95))
    else:
        max_document_length = max(post_length)
    print("Document length : " + str(max_document_length))

    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, MAX_FEATURES)
    vocab_processor = vocab_processor.fit(x_text)

    trainX = np.array(list(vocab_processor.transform(X_train)))
    testX = np.array(list(vocab_processor.transform(X_test)))

    trainY = np.asarray(y_train)
    testY = np.asarray(y_test)

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)

    trainY = to_categorical(trainY, nb_classes=NUM_CLASSES)
    testY = to_categorical(testY, nb_classes=NUM_CLASSES)

    data_dict = {
            "data": data,
            "trainX" : trainX,
            "trainY" : trainY,
            "testX" : testX,
            "testY" : testY,
            "vocab_processor" : vocab_processor
        }
    return data_dict
    data = pickle.load(open(get_filename("twitter"), 'rb'))
    x_text=[]
    labels=[]
    x_text_train = []
    labels_train = [] 
    x_text_test = []
    labels_test = [] 
    
    dict1 = {'racism':2,'sexism':1,'none':0}
    for i in data:
        labels.append(i['label'])
        x_text.append(i['text'])       

        
    labels = [dict1[b] for b in labels]

    print("Counter before oversampling ALL")
    from collections import Counter
    print(Counter(labels))
    
    racism = [i for i in range(len(labels)) if labels[i]==2]
    sexism = [i for i in range(len(labels)) if labels[i]==1]
    
    print(len(racism))
    print(len(sexism))
    
    data = data + [data[x] for x in racism]*(oversampling_rate-1)+ [data[x] for x in sexism]*(oversampling_rate-1)
    labels = labels + [2 for i in range(len(racism))]*(oversampling_rate-1) + [1 for i in range(len(sexism))]*(oversampling_rate-1)
     
    print("Counter AFTER oversampling ALL")
    from collections import Counter
    print(Counter(labels))
    
    data_dicts=[]
    for test_users in particion1:
        for i in range(len(data)):
            if data[i]['user'] not in test_users:
                x_text_train.append(data[i]['text'])
                labels_train.append(labels[i])
            else:
                x_text_test.append(data[i]['text'])
                labels_test.append(labels[i])
        
        NUM_CLASSES = 3
        print("Counter TRAIN")
        from collections import Counter
        print(Counter(labels_train))

        print("Counter test")
        from collections import Counter
        print(Counter(labels_test))

        post_length = np.array([len(x.split(" ")) for x in x_text])
        if(data != "twitter"):
            max_document_length = int(np.percentile(post_length, 95))
        else:
            max_document_length = max(post_length)
        print("Document length : " + str(max_document_length))

        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, MAX_FEATURES)
        vocab_processor = vocab_processor.fit(x_text)

        trainX = np.array(list(vocab_processor.transform(x_text_train)))
        testX = np.array(list(vocab_processor.transform(x_text_test)))

        trainY = np.asarray(labels_train)
        testY = np.asarray(labels_test)

        trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
        testX = pad_sequences(testX, maxlen=max_document_length, value=0.)

        trainY = to_categorical(trainY, nb_classes=NUM_CLASSES)
        testY = to_categorical(testY, nb_classes=NUM_CLASSES)

        data_dict = {
            "data": data,
            "trainX" : trainX,
            "trainY" : trainY,
            "testX" : testX,
            "testY" : testY,
            "vocab_processor" : vocab_processor
        }
                
        data_dicts.append(data_dict)
    return data_dicts

y_test=[]

dict_total = {}
def get_users_behavior(files):
    #cantidad de tweets por usuario en cada clase
    tweets = []
    dict_users_normal = {}
    dict_users_hate = {}
    total_tweets=0
    normal = 0
    hate =0
    for file in files:
        data = pd.read_csv(file,'r',delimiter=',')
        for line in data.values:
            y_test.append(line[0]) 
            if line[0] == 0:
                total_tweets +=1
                normal +=1
                if line[3] in dict_users_normal.keys():
                    dict_users_normal[line[3]] += 1
                else:
                    dict_users_normal[line[3]] = 1
                if line[3] in dict_total.keys():
                    dict_total[line[3]] += 1
                else:
                    dict_total[line[3]] = 1
            else:
                total_tweets +=1
                hate+=1
                if line[3] in dict_users_hate.keys():
                    dict_users_hate[line[3]] += 1
                else:
                    dict_users_hate[line[3]] = 1 
                if line[3] in dict_total.keys():
                    dict_total[line[3]] += 1
                else:
                    dict_total[line[3]] = 1
    print('Usuarios Normales') 
    print(total_tweets)
    print(normal)
    print(hate)
    print('\n') 
    print('Usuarios Normales') 
    print(len(dict_users_normal))
    resultado = sorted(dict_users_normal.items(), key=operator.itemgetter(1))
    resultado.reverse()
    print (resultado[:10])
    print('\n') 
    print('Usuarios Hateful') 
    print(len(dict_users_hate))
    resultado = sorted(dict_users_hate.items(), key=operator.itemgetter(1))
    resultado.reverse()
    print (resultado[:10])
    print('\n') 
    print('Todos los usuarios') 
    print(len(dict_total))
    resultado = sorted(dict_total.items(), key=operator.itemgetter(1))
    resultado.reverse()
   # print (resultado)

def print_detalle_particion(X_train,y_train,X_test,y_test):
    dict_users_train = {}
    sexist_train=0
    racist_train=0
    normal_train=0
    dict_users_test = {}
    sexist_test=0
    normal_test=0
    racist_test=0
    for i in X_train:
        if i in dict_users_train.keys():
            dict_users_train[i] += 1
        else:
            dict_users_train[i] = 1
    for i in y_train:
        if i == 0:
            normal_train += 1
        elif i == 2:
            sexist_train += 1
        else:
            racist_train += 1  

    for i in X_test:
        if i in dict_users_test.keys():
            dict_users_test[i] += 1
        else:
            dict_users_test[i] = 1
    for i in y_test:
        if i == 0:
            normal_test += 1
        elif i == 2:
            sexist_test += 1
        else:
            racist_test += 1  
            
    print('En TRAIN SET')
    print('Cantidad de Usuarios: '+ str(len(dict_users_train))) 
    print('Tweets sexist: '+ str(sexist_train)) 
    print('Tweets racist: '+ str(racist_train)) 
    print('Tweets Normales: '+ str(normal_train)) 
    print('\n')
    resultado = sorted(dict_users_train.items(), key=operator.itemgetter(1))
    resultado.reverse()
    #print(resultado[0:10])
    print('En TEST SET')
    print('Cantidad de Usuarios: '+ str(len(dict_users_test))) 
    print('Tweets Odiosos: '+ str(hate_test)) 
    print('Tweets Normales: '+ str(normal_test)) 
    print('\n')
    resultado = sorted(dict_users_test.items(), key=operator.itemgetter(1))
    resultado.reverse()
    #print(resultado[0:10])
    
def Tweets_users_hateful(tweets): #seleccionando para training los usuarios mas prolíficos
    hateful_tweets = []
    test_index = []
    clasestrain = {}
    clasestest ={}
    hateful_users = ['Levi Stein',"Yes You're Sexist",'Male Tears #4648']#tres usuarios mas odiosos
    for tweet in tweets:
        if tweet['name'] not in hateful_users:            
            hateful_tweets.append(tweet)
        else:
            pass
    print('len(hateful_tweets)')
    print(len(hateful_tweets))   
    return hateful_tweets
    
def Holdout_partition(data): #seleccionando para training los usuarios mas prolíferos
    train_index = []
    test_index = []
    clasestrain = {}
    clasestest ={}
    train_users = ['Levi Stein',"Yes You're Sexist",'Male Tears #4648']#tres usuarios mas odiosos
    X_train_users = []
    X_test_users = []
    for i in range(len(data)):
        if data[i]['name'] in train_users:            
            train_index.append(i)
            X_train_users.append(data[i]['name'])
            if data[i]['label'] in clasestrain.keys():
                    clasestrain[data[i]['label']] += 1
            else:
                    clasestrain[data[i]['label']] = 1
            #print(data[i]['name'])
        else:
            test_index.append(i)
            X_test_users.append(data[i]['name'])
            if data[i]['label'] in clasestest.keys():
                clasestest[data[i]['label']] += 1
            else:
                clasestest[data[i]['label']] = 1
    #print_detalle_particion(X_train_users,y_train,X_test_users,y_test)            
    print('train')
    print(clasestrain.keys())
    print('normales:' +str(clasestrain['neither.json']))
    print('racistas:' +str(clasestrain['racism.json']))
    print('sexistas:' +str(clasestrain['sexism.json']))
    print('test')
    print('normales:' +str(clasestest['neither.json']))
    print('racistas:' +str(clasestest['racism.json']))
    print('sexistas:' +str(clasestest['sexism.json']))
    print(len(train_index))
    print(len(test_index))   
    
    print(clasestrain)
    print(clasestest)
    return train_index, test_index

def user_kfold():
    data = pd.read_csv('DatosCSV\en_total.csv','r',delimiter=',')
    import numpy as np
    X=[]#usuarios
    y=[]#clase
    for line in data.values:
        y.append(line[0]) 
        X.append(line[3])
    X=np.array(X)
    y=np.array(y)

    from sklearn.model_selection import KFold
    KFold(n_splits=3, random_state=0, shuffle=False)
    kf = KFold(n_splits=3)
    kf.get_n_splits(X)
    p=0
    TESTS = []
    for train_index, test_index in kf.split(X,y):
        p += 1
        print('Particion' + str(p))
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #print_detalle_particion(X_train, y_train,X_test, y_test)
        print(len(X_train),len(X_test))
        TESTS.append((X_train,X_test))
        for i in X_train:
            if i in X_test:
                print('Usuario Solapado:' + str(i))
        print('TESTS.shape()')
        print(TESTS[0])
    return TESTS

def get_splits(tweets,trList, tsList):
    #user_split = user_kfold()
    i=0
    sets=[]
    train_index,test_index=[],[]
    y_map = {
        'neither.json': 0,
        'racism.json': 1,
        'sexism.json': 2
        }
    print('len(trList), len(tsList)')
    print(len(trList), len(tsList))
    for tweet in tweets:
        #print(tweet)
        text = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(tweet['text'].lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        seq, _emb = [], []
        for word in words:
            seq.append(vocab.get(word, vocab['UNK']))
        if tweet['name'] in tsList:
            test_index.append(i)
        elif tweet['name'] in trList:
            train_index.append(i)
        else:
            pass
        i+=1
    return train_index, test_index


