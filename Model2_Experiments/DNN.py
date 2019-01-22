from models import get_model
import argparse
import pickle
import string
import numpy as np
from sklearn.model_selection import train_test_split, KFold
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
from auxiliares import *

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

def load_data_old(filename):
    print("Loading data from file: " + filename)
    data = pickle.load(open(filename, 'rb'))
    x_text = []
    labels = [] 
    for i in range(len(data)):
        if(HASH_REMOVE):
            x_text.append(data[i]['text'])
        else:
            x_text.append(data[i]['text'])
        labels.append(data[i]['label'])
    return x_text,labels

def get_filename(dataset):
    global NUM_CLASSES, HASH_REMOVE
    if(dataset=="twitter"):
        NUM_CLASSES = 3
        HASH_REMOVE = True
        filename = "DatosCSV/twitter_data.pkl"
    elif(dataset=="formspring"):
        NUM_CLASSES = 2
        filename = "data/formspring_data.pkl"
    elif(dataset=="wiki"):
        NUM_CLASSES = 2
        filename = "data/wiki_data.pkl"
    return filename

def load_data(filename,flag):
    if flag =='train':
        print("Loading data from file: " + filename)
        data = pickle.load(open('DatosCSV/twitter_data.pkl', 'rb'))
               
    elif flag == 'test':
        print("Loading data from file: " + filename)
        #data = pickle.load(open('DatosCSV/twitter_data_davidson.pkl', 'rb'))    
        data = pickle.load(open('DatosCSV/sem_eval.pkl', 'rb'))   
    elif flag == 'data_new':
        print("Loading data from file: " + filename)
        data = pickle.load(open('DatosCSV/data_nueva.pkl', 'rb')) 
        data=[]
    x_text = []
    labels = [] 
    for i in range(len(data)):
        if(HASH_REMOVE):
            x_text.append(data[i]['text'])
        else:
            x_text.append(data[i]['text'])
        labels.append(data[i]['label'])
    return x_text,labels

def get_data_old(data, oversampling_rate):
    
    x_text, labels = load_data_old(get_filename(data)) 
 
    if(data=="twitter"):
        NUM_CLASSES = 3
        dict1 = {'racism':2,'sexism':1,'none':0}
        labels = [dict1[b] for b in labels]
        
        racism = [i for i in range(len(labels)) if labels[i]==2]
        sexism = [i for i in range(len(labels)) if labels[i]==1]
        x_text = x_text + [x_text[x] for x in racism]*(oversampling_rate-1)+ [x_text[x] for x in sexism]*(oversampling_rate-1)
        labels = labels + [2 for i in range(len(racism))]*(oversampling_rate-1) + [1 for i in range(len(sexism))]*(oversampling_rate-1)
    
    else:  
        NUM_CLASSES = 2
        bully = [i for i in range(len(labels)) if labels[i]==1]
        x_text = x_text + [x_text[x] for x in bully]*(oversampling_rate-1)
        labels = list(labels) + [1 for i in range(len(bully))]*(oversampling_rate-1)

    print("Counter after oversampling")
    from collections import Counter
    print(Counter(labels))
    
    filter_data = []
    for text in x_text:
       # print(text.type)
        filter_data.append("".join(l for l in text if l not in string.punctuation))
        
    return x_text, labels
def get_filename(dataset):
    global NUM_CLASSES, HASH_REMOVE
    if(dataset=="twitter"):
        NUM_CLASSES = 3
        HASH_REMOVE = True
        filename = "DatosCSV/twitter_data.pkl"
    elif(dataset=="formspring"):
        NUM_CLASSES = 2
        filename = "data/formspring_data.pkl"
    elif(dataset=="wiki"):
        NUM_CLASSES = 2
        filename = "data/wiki_data.pkl"
    return filename

def get_embedding_weights(filename, sep):
    embed_dict = {}
    file = open(filename,'r', encoding="utf8")
    for line in file.readlines():
        row = line.strip().split(sep)
        embed_dict[row[0]] = row[1:]
    print('Loaded from file: ' + str(filename))
    file.close()
    return embed_dict

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

def evaluate_model(model, testX, testY):
    temp = model.predict(testX)
     
    y_pred  = np.argmax(temp, 1)

    y_true = np.argmax(testY, 1)
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

def dump_learned_embedding(data, model_type, vector_type, embed_size, embed, vocab_processor):
    vocab = vocab_processor.vocabulary_._mapping
    vocab_size = len(vocab)
    embedDict = {}
    n = 0
    words_missed = []
    for k, v in vocab.items():
        try:
            embeddingDict[v] = embed[k]
        except:
            n += 1
            words_missed.append(k)
            pass
    print("%d embedding missed"%n, " of " , vocab_size)
    
    filename = output_folder_name + data + "_" + model_type + "_" + vector_type + "_" + embed_size + ".pkl"
    with open(filename, 'wb',encoding='utf-8') as handle:
        pickle.dump(embedDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_train_test(data, x_text, labels):
    
    X_train, X_test, Y_train, Y_test = train_test_split(x_text, labels, random_state=42, test_size=0.10)
    
    print(len(X_train))
    print(len(X_test))
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
    
    trainY = np.asarray(Y_train)
    testY = np.asarray(Y_test)
        
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

def return_data(data_dict):
    return data_dict["data"], data_dict["trainX"], data_dict["trainY"], data_dict["testX"], data_dict["testY"],data_dict["vocab_processor"]

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
    
def train(data_dict, model_type, vector_type, embed_size, dump_embeddings=False):
    print("trainnn")   
    data, trainX, trainY, testX, testY, vocab_processor = return_data(data_dict)
    
    vocab_size = len(vocab_processor.vocabulary_)
    print("Vocabulary Size: {:d}".format(vocab_size))
    vocab = vocab_processor.vocabulary_._mapping
    
    print("Running Model: " + model_type + " with word vector initiliazed with " + vector_type + " word vectors.")
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
            print("Word vectors used: " + vector_type)
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
    
    
    return  evaluate_model(model, testX, testY)

def print_scores(precision_scores, recall_scores, f1_scores):
    for i in range(NUM_CLASSES):
        print("\nPrecision Class %d (avg): %0.3f (+/- %0.3f)" % (i, precision_scores[:, i].mean(), precision_scores[:, i].std() * 2))
        print( "\nRecall Class %d (avg): %0.3f (+/- %0.3f)" % (i, recall_scores[:, i].mean(), recall_scores[:, i].std() * 2))
        print( "\nF1 score Class %d (avg): %0.3f (+/- %0.3f)" % (i, f1_scores[:, i].mean(), f1_scores[:, i].std() * 2))
        
def get_data_OLD(data, oversampling_rate):
    
    x_text, labels = load_data_old(get_filename(data)) 
    print("Counter before oversampling")
    from collections import Counter
    print(Counter(labels))
    
    if(data=="twitter"):
        NUM_CLASSES = 3
        dict1 = {'racism':2,'sexism':1,'none':0}
        labels = [dict1[b] for b in labels]
        
        racism = [i for i in range(len(labels)) if labels[i]==2]
        sexism = [i for i in range(len(labels)) if labels[i]==1]
        x_text = x_text + [x_text[x] for x in racism]*(oversampling_rate-1)+ [x_text[x] for x in sexism]*(oversampling_rate-1)
        labels = labels + [2 for i in range(len(racism))]*(oversampling_rate-1) + [1 for i in range(len(sexism))]*(oversampling_rate-1)
    
    else:  
        NUM_CLASSES = 2
        bully = [i for i in range(len(labels)) if labels[i]==1]
        x_text = x_text + [x_text[x] for x in bully]*(oversampling_rate-1)
        labels = list(labels) + [1 for i in range(len(bully))]*(oversampling_rate-1)

    print("Counter after oversampling")
    from collections import Counter
    print(Counter(labels))
    
    filter_data = []
    for text in x_text:
       # print(text.type)
        filter_data.append("".join(l for l in text if l not in string.punctuation))
     
    return x_text, labels

def get_data(data, oversampling_rate,flag):
    x_text, labels = load_data(get_filename(data),flag)
    print("Counter beforeeeee oversampling")
    from collections import Counter
    print(Counter(labels))
    if(data=="twitter" and flag == 'train'):
        NUM_CLASSES = 3
        dict1 = {'racism':1,'sexism':2,'none':0,'hate':1}
        labels = [dict1[b] for b in labels]
        
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
    
    if(data=="twitter" and flag == 'data_new'):
        NUM_CLASSES = 2
        dict1 = {'sexism':2,'racism':1,'none':0,'hate':1}
        labels = [dict1[b] for b in labels]
        
        racism = [i for i in range(len(labels)) if labels[i]==2]
        sexism = [i for i in range(len(labels)) if labels[i]==1]
        x_text = x_text + [x_text[x] for x in racism]*(oversampling_rate-1)+ [x_text[x] for x in sexism]*(oversampling_rate-1)
        labels = labels + [2 for i in range(len(racism))]*(oversampling_rate-1) + [1 for i in range(len(sexism))]*(oversampling_rate-1)

        print("Counter afterrrrr oversampling")
        from collections import Counter
        print(Counter(labels))
        
    elif (data=="twitter" and flag == 'test'):
        NUM_CLASSES = 2
        dict1 = {'hate':1,'none':0}
        labels = [dict1[b] for b in labels]
        print("Counter afterrrrr oversampling")
        from collections import Counter
        print(Counter(labels))
        
    elif data !="twitter":  
        NUM_CLASSES = 2
        bully = [i for i in range(len(labels)) if labels[i]==1]
        x_text = x_text + [x_text[x] for x in bully]*(oversampling_rate-1)
        labels = list(labels) + [1 for i in range(len(bully))]*(oversampling_rate-1)
        
        print("Counter afterrrrr oversampling")
        from collections import Counter
        print(Counter(labels))

    print("Counter afterrrrr oversampling")
    from collections import Counter
    print(Counter(labels))
    
    filter_data = []
    i=0
    for text in x_text:
        try:
            filter_data.append("".join(l for l in text if l not in string.punctuation))
        except Exception as e:
            print(i)
            print(text)
        i += 1

    return x_text, labels

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

def run_model_exp1(data, oversampling_rate, model_type, vector_type, embed_size):    
    x_text, labels = get_data(data, 3,'train')
    cv_object = KFold(n_splits=5, shuffle=True, random_state=42)
    a, p, r, f1 = 0., 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    pn, rn, fn = 0., 0., 0.
    post_length = np.array([len(x.split(" ")) for x in x_text])
    if(data != "twitter"):
        max_document_length = int(np.percentile(post_length, 95))
    else:
        max_document_length = max(post_length)
    print("Document length : " + str(max_document_length))

    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, MAX_FEATURES)
    vocab_processor = vocab_processor.fit(x_text) 
    
    for train_index, test_index in cv_object.split(x_text):
        X_train, y_train, X_test, y_test = [],[],[],[]
        for i in range(len(x_text)):
            if i in train_index:
                X_train.append(x_text[i])
                y_train.append(labels[i])
            else:
                X_test.append(x_text[i])
                y_test.append(labels[i])
        

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
        
        precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem = train(data_dict, model_type, vector_type, embed_size)    

        #a += accuracy_score (y_test, y_pred)
        p += precisionw#_score(y_test, y_pred, average='weighted')
        p1 += precisionm#_score(y_test, y_pred, average='micro')
        r += recallw#_score(y_test, y_pred, average='weighted')
        r1 += recallm#_score(y_test, y_pred, average='micro')
        f1 += f1_scorew#(y_test, y_pred, average='weighted')
        f11 += f1_scorem#(y_test, y_pred, average='micro')
        pn += precision
        rn += recall
        fn += f1_score
        
    NO_OF_FOLDS = 5
    print ("None results are")
    print (prod(pn, NO_OF_FOLDS))
    print (prod(rn, NO_OF_FOLDS))
    print (prod(fn, NO_OF_FOLDS))    
        
    print ("weigthed results are")
    print ("average accuracy is %f" %(a/NO_OF_FOLDS))
    print ("average precision is %f" %(p/NO_OF_FOLDS))
    print ("average recall is %f" %(r/NO_OF_FOLDS))
    print ("average f1 is %f" %(f1/NO_OF_FOLDS))

    print ("micro results are")
    print ("average precision is %f" %(p1/NO_OF_FOLDS))
    print ("average recall is %f" %(r1/NO_OF_FOLDS))
    print ("average f1 is %f" %(f11/NO_OF_FOLDS))

def run_model_exp10(data, oversampling_rate, model_type, vector_type, embed_size):    
    x_text, labels = load_data(get_filename(data),'train')
    
    cv_object = KFold(n_splits=5, shuffle=True, random_state=42)
    a, p, r, f1 = 0., 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    pn, rn, fn = 0., 0., 0.
    
    post_length = np.array([len(x.split(" ")) for x in x_text])
    if(data != "twitter"):
        max_document_length = int(np.percentile(post_length, 95))
    else:
        max_document_length = max(post_length)
    print("Document length : " + str(max_document_length))

    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, MAX_FEATURES)
    vocab_processor = vocab_processor.fit(x_text) 
    
    for train_index, test_index in cv_object.split(x_text):
        X_train, y_train, X_test, y_test = [],[],[],[]
        for i in range(len(x_text)):
            if i in train_index:
                X_train.append(x_text[i])
                y_train.append(labels[i])
            else:
                X_test.append(x_text[i])
                y_test.append(labels[i])
        
                NUM_CLASSES = 3
            
        dict1 = {'sexism':2,'racism':1,'none':0}
        y_train = [dict1[b] for b in y_train]
        y_test = [dict1[b] for b in y_test]
        
        print("Counter before oversampling")
        from collections import Counter
        print(Counter(y_train))
        #oversampling training set
        racism = [i for i in range(len(y_train)) if y_train[i]==2]
        sexism = [i for i in range(len(y_train)) if y_train[i]==1]
        X_train = X_train + [X_train[x] for x in racism]*(oversampling_rate-1)+ [X_train[x] for x in sexism]*(oversampling_rate-1)
        y_train = y_train + [2 for i in range(len(racism))]*(oversampling_rate-1) + [1 for i in range(len(sexism))]*(oversampling_rate-1)

        print("Counter afterrrrr oversampling")
        from collections import Counter
        print(Counter(y_train))

    
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
        
        precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem = train(data_dict, model_type, vector_type, embed_size)    

        p += precisionw
        p1 += precisionm
        r += recallw
        r1 += recallm
        f1 += f1_scorew
        f11 += f1_scorem
        pn += precision
        rn += recall
        fn += f1_score
    NO_OF_FOLDS = 5
    print ("None results are")
    print (prod(pn, NO_OF_FOLDS))
    print (prod(rn, NO_OF_FOLDS))
    print (prod(fn, NO_OF_FOLDS))    
        
    print ("weigthed results are")
    print ("average accuracy is %f" %(a/NO_OF_FOLDS))
    print ("average precision is %f" %(p/NO_OF_FOLDS))
    print ("average recall is %f" %(r/NO_OF_FOLDS))
    print ("average f1 is %f" %(f1/NO_OF_FOLDS))

    print ("micro results are")
    print ("average precision is %f" %(p1/NO_OF_FOLDS))
    print ("average recall is %f" %(r1/NO_OF_FOLDS))
    print ("average f1 is %f" %(f11/NO_OF_FOLDS))
    
    
def run_model_exp2(data, oversampling_rate, model_type, vector_type, embed_size):    

    data_dict = Holdout_partition(data,oversampling_rate)
    a, p, r, f1 = 0., 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    pn, rn, fn = 0., 0., 0.
    
    precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem = train(data_dict, model_type, vector_type, embed_size)    
  
    print('RESULTS EXPERIMENTO 2')
    print ("NONE AVERAGE results are")
    print (precision)
    print (recall)
    print (f1_score)    
        
    print ("WEIGTHED AVERAGE results are")
    print ("average precision is %f" %(precisionw))
    print ("average recall is %f" %(recallw))
    print ("average f1 is %f" %(f1_scorew))

    print ("micro results are")
    print ("average precision is %f" %precisionm)
    print ("average recall is %f" %recallm)
    print ("average f1 is %f" %f1_scorem)


def run_model_exp3(data, oversampling_rate, model_type, vector_type, embed_size):    
    data_dicts = user_kfold("twitter", 3)
    a, p, r, f1 = 0., 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    pn, rn, fn = 0., 0., 0.
    for data_dict in data_dicts:
        precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem = train(data_dict, model_type, vector_type, embed_size)    

        #a += accuracy_score (y_test, y_pred)
        p += precisionw#_score(y_test, y_pred, average='weighted')
        p1 += precisionm#_score(y_test, y_pred, average='micro')
        r += recallw#_score(y_test, y_pred, average='weighted')
        r1 += recallm#_score(y_test, y_pred, average='micro')
        f1 += f1_scorew#(y_test, y_pred, average='weighted')
        f11 += f1_scorem#(y_test, y_pred, average='micro')
        pn += precision
        rn += recall
        fn += f1_score
    NO_OF_FOLDS = 3
    print ("None results areeeeeeeeeeeeeeeeeeeee")
    print (prod(pn, NO_OF_FOLDS))
    print (prod(rn, NO_OF_FOLDS))
    print (prod(fn, NO_OF_FOLDS))    
        
    print ("weigthed results areeeeeeeeeeeeeeeeeeeee")
    print ("average accuracy is %f" %(a/NO_OF_FOLDS))
    print ("average precision is %f" %(p/NO_OF_FOLDS))
    print ("average recall is %f" %(r/NO_OF_FOLDS))
    print ("average f1 is %f" %(f1/NO_OF_FOLDS))

    print ("micro results are")
    print ("average precision is %f" %(p1/NO_OF_FOLDS))
    print ("average recall is %f" %(r1/NO_OF_FOLDS))
    print ("average f1 is %f" %(f11/NO_OF_FOLDS))    
    

def run_model_exp4(data, oversampling_rate, model_type, vector_type, embed_size): 
    X_train1, y_train1 = get_data(data, oversampling_rate,'train')
    print(len('X_train1m;fmf;ls;lsk'))
    print(len(X_train1))
    print(len(y_train1))
    X_train = []
    y_train=[]
    for i in range(len(X_train1)):
        if i != 24782:
            X_train.append(X_train1[i])
            y_train.append(y_train1[i])
        else:
            print(X_train1[i])
    
    X_test, y_test = get_data(data, oversampling_rate,'test')
    
    data_dict = get_data_dict(X_train,X_test,y_train,y_test)
    a, p, r, f1 = 0., 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    pn, rn, fn = 0., 0., 0.
    precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem = train(data_dict, model_type, vector_type, embed_size)    

    #a += accuracy_score (y_test, y_pred)
    p += precisionw
    p1 += precisionm
    r += recallw
    r1 += recallm
    f1 += f1_scorew
    f11 += f1_scorem
    pn += precision
    rn += recall
    fn += f1_score
    print ("None Average results are:")
    print (pn)
    print (rn)
    print (fn)    
        
    print ("weigthed results are:")
    print ("average accuracy is %f" %(a))
    print ("average precision is %f" %(p))
    print ("average recall is %f" %(r))
    print ("average f1 is %f" %(f1))

    print ("micro results are")
    print ("average precision is %f" %(p1))
    print ("average recall is %f" %(r1))
    print ("average f1 is %f" %(f11))    

def run_model(data, oversampling_rate, model_type, vector_type, embed_size):    
    
    a, p, r, f1 = 0., 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    pn, rn, fn = 0., 0., 0.
    x_text, labels = get_data_old(data, oversampling_rate)
    #print(x_text[0:10])
    data_dict = get_train_test(data,  x_text, labels)
    print("train Counter before oversampling")
    data, trainX, trainY, testX, testY, vocab_processor = return_data(data_dict)
    print(trainY)
    print(testY)
    print(trainY[0])
    precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem = train_old(data_dict, model_type, vector_type, embed_size)
    #a += accuracy_score (y_test, y_pred)
    p += precisionw#_score(y_test, y_pred, average='weighted')
    p1 += precisionm#_score(y_test, y_pred, average='micro')
    r += recallw#_score(y_test, y_pred, average='weighted')
    r1 += recallm#_score(y_test, y_pred, average='micro')
    f1 += f1_scorew#(y_test, y_pred, average='weighted')
    f11 += f1_scorem#(y_test, y_pred, average='micro')
    pn += precision
    rn += recall
    fn += f1_score
    NO_OF_FOLDS = 1
    print ("None results areeeeeeeeeeeeeeeeeeeee")
    print (prod(pn, NO_OF_FOLDS))
    print (prod(rn, NO_OF_FOLDS))
    print (prod(fn, NO_OF_FOLDS))    
        
    print ("weigthed results areeeeeeeeeeeeeeeeeeeee")
    print ("average accuracfy is %f" %(a/NO_OF_FOLDS))
    print ("average precision is %f" %(p/NO_OF_FOLDS))
    print ("average recall is %f" %(r/NO_OF_FOLDS))
    print ("average f1 is %f" %(f1/NO_OF_FOLDS))

    print ("micro results are")
    print ("average precision is %f" %(p1/NO_OF_FOLDS))
    print ("average recall is %f" %(r1/NO_OF_FOLDS))
    print ("average f1 is %f" %(f11/NO_OF_FOLDS)) 
    
if __name__ == "__main__":
    print('main')
    data = "twitter"
    model_type = "blstm"
    vector_type = "sswe"

    for embed_size in [50]:#[25, 50, 100, 200]:
        run_model_exp10(data,3 ,model_type, vector_type, embed_size)
   