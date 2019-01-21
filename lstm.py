from data_handler import get_data
import argparse
import keras
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, LSTM
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Convolution1D, MaxPooling1D, GlobalMaxPooling1D
import numpy as np
import pdb
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
from auxiliares import *
from nltk import tokenize as tokenize_nltk
from my_tokenizer import glove_tokenize
import json


### Preparing the text data
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}



EMBEDDING_DIM = None
GLOVE_MODEL_FILE = None
SEED = 42
NO_OF_FOLDS = 10
CLASS_WEIGHT = None
LOSS_FUN = None
OPTIMIZER = None
KERNEL = None
TOKENIZER = None
MAX_SEQUENCE_LENGTH = None
INITIALIZE_WEIGHTS_WITH = None
LEARN_EMBEDDINGS = None
EPOCHS = 10
BATCH_SIZE = 512
SCALE_LOSS_FUN = None

word2vec_model = None

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


def get_embedding(word):
    #return
    try:
        return word2vec_model[word]
    except Exception as e:
        print ('Encoding not found: %s' %(word))
        return np.zeros(EMBEDDING_DIM)

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
    return embedding



def select_tweets():
    # selects the tweets as in mean_glove_embedding method
    # Processing
    tweets = get_data()
    print('len(tweets)')
    print(len(tweets[0]))
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
    return tweet_return


def gen_vocab():
    # Processing
    vocab_index = 1
    for tweet in tweets:
        text = TOKENIZER(tweet['text'].lower())
        text = ' '.join([c for c in text if c not in punctuation])
        #print(text)
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
    print('len(vocab)')
    print(len(vocab))


def filter_vocab(k):
    global freq, vocab
    pdb.set_trace()
    freq_sorted = sorted(freq.items(), key=operator.itemgetter(1))
    tokens = freq_sorted[:k]
    vocab = dict(zip(tokens, range(1, len(tokens) + 1)))
    vocab['UNK'] = len(vocab) + 1
    print('ya')


def gen_sequence(tweets):
    y_map = {
            'neither.json': 0,
            'racism.json': 1,
            'sexism.json': 2
            }

    X, y = [], []
    for tweet in tweets:
        #print(tweet)
        text = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(tweet['text'].lower())
        text = ' '.join([c for c in text if c not in punctuation])
        #print(text)
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        #print(len(words))
        seq, _emb = [], []
        for word in words:
            seq.append(vocab.get(word, vocab['UNK']))
            #print(vocab.get(word))#, vocab['UNK']))
        #print(len(seq))
        X.append(seq)
        #print(len(X[0]))
        y.append(y_map[tweet['label']])
    print('len(vocab)')
    print(len(vocab))

    return X, y


def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)

def lstm_model(sequence_length, embedding_dim):
    model_variation = 'LSTM'
    print('Model variation is %s' % model_variation)
    model = Sequential()
    print('variables')
    print(embedding_dim)
    print(sequence_length)
    print(len(vocab))
    model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length, trainable=LEARN_EMBEDDINGS))
    model.add(Dropout(0.25))#, input_shape=(sequence_length, embedding_dim)))
    model.add(LSTM(embedding_dim))#, input_shape=(sequence_length, embedding_dim)))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss=LOSS_FUN, optimizer=OPTIMIZER, metrics=['accuracy'])
    print (model.summary())
    return model


def train_LSTM_Holdout(X,y, model, inp_dim, weights, train_index, test_index, epochs=EPOCHS, batch_size=BATCH_SIZE):
    
    a, p, r, f1 = 0., 0., 0., 0.
    a1, p1, r1, f11 = 0., 0., 0., 0.
    sentence_len = X.shape[1]
    if INITIALIZE_WEIGHTS_WITH == "glove":
        model.layers[0].set_weights([weights])
    elif INITIALIZE_WEIGHTS_WITH == "random":
        shuffle_weights(model)
    else:
        print ("ERROR!")
        return
    
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    
    
    y_train = y_train.reshape((len(y_train), 1))
    X_temp = np.hstack((X_train, y_train))
    for epoch in range(epochs):
        for X_batch in batch_gen(X_temp, batch_size):
            x = X_batch[:, :sentence_len]
            y_temp = X_batch[:, sentence_len]

            class_weights = None
            if SCALE_LOSS_FUN:
                class_weights = {}
                class_weights[0] = np.where(y_temp == 0)[0].shape[0]/float(len(y_temp))
                class_weights[1] = np.where(y_temp == 1)[0].shape[0]/float(len(y_temp))
                class_weights[2] = np.where(y_temp == 2)[0].shape[0]/float(len(y_temp))

            try:
                y_temp = np_utils.to_categorical(y_temp, num_classes=3)
            except Exception as e:
                print (e)
                print (y_temp)
            loss, acc = model.train_on_batch(x, y_temp, class_weight=class_weights)

    y_pred = model.predict_on_batch(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    print (classification_report(y_test, y_pred))
    print (precision_recall_fscore_support(y_test, y_pred))
    #print (y_pred)
    a = accuracy_score (y_test, y_pred)
    p = precision_score(y_test, y_pred, average='weighted')
    p1 = precision_score(y_test, y_pred, average='micro')
    r = recall_score(y_test, y_pred, average='weighted')
    r1 = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='weighted')
    f11 = f1_score(y_test, y_pred, average='micro')

    print ("macro results areeeeeeeeeeeeeeeeeeeee")
    print ("average accuracy is %f" %(a))
    print ("average precision is %f" %(p1))
    print ("average recall is %f" %(r1))
    print ("average f1 is %f" %(f1))

def train_LSTM_userKF(X,y,model, inp_dim, weights, epochs=EPOCHS, batch_size=BATCH_SIZE):
    a, p, r, f1 = 0., 0., 0., 0.
    a1, p1, r1, f11 = 0., 0., 0., 0.
    i=0
    sentence_len = X.shape[1]
    aasa = user_kfold()
    for trList, tsList in aasa:
        train_index, test_index = get_splits(tweets,trList, tsList) 
        if INITIALIZE_WEIGHTS_WITH == "glove":
            model.layers[0].set_weights([weights])
        elif INITIALIZE_WEIGHTS_WITH == "random":
            shuffle_weights(model)
        else:
            print ("ERROR!")
            return
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        
        print('len(train_index), len(test_index)')
        print(len(train_index), len(test_index ))
        print(len(X_train), len(y_train ))
        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train))
        for epoch in range(epochs):
            for X_batch in batch_gen(X_temp, batch_size):
                x = X_batch[:, :sentence_len]
                y_temp = X_batch[:, sentence_len]

                class_weights = None
                if SCALE_LOSS_FUN:
                    class_weights = {}
                    class_weights[0] = np.where(y_temp == 0)[0].shape[0]/float(len(y_temp))
                    class_weights[1] = np.where(y_temp == 1)[0].shape[0]/float(len(y_temp))
                    class_weights[2] = np.where(y_temp == 2)[0].shape[0]/float(len(y_temp))

                try:
                    y_temp = np_utils.to_categorical(y_temp, num_classes=3)
                except Exception as e:
                    print (e)
                    #print (y_temp)
                #print (x.shape, y_temp.shape)
                loss, acc = model.train_on_batch(x, y_temp, class_weight=class_weights)
                #print (loss, acc)
        if i ==0:
            weights = model.layers[0].get_weights()[0]
        else:
            weights += model.layers[0].get_weights()[0]
        i=i+1
        print('weights.shape')
        print(weights.shape)
        y_pred = model.predict_on_batch(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        print (classification_report(y_test, y_pred))
        print (precision_recall_fscore_support(y_test, y_pred))
        #print (y_pred)
        a += accuracy_score (y_test, y_pred)
        p += precision_score(y_test, y_pred, average='weighted')
        p1 += precision_score(y_test, y_pred, average='micro')
        r += recall_score(y_test, y_pred, average='weighted')
        r1 += recall_score(y_test, y_pred, average='micro')
        f1 += f1_score(y_test, y_pred, average='weighted')
        f11 += f1_score(y_test, y_pred, average='micro')
    
    NO_OF_FOLDS = 3
    np.save('lstm.npy',weights/NO_OF_FOLDS)
    ff=open('vocab','wb')

    print ("macro results areeeeeeeeeeeeeeeeeeeee")
    print ("average accuracy is %f" %(a/NO_OF_FOLDS))
    print ("average precision is %f" %(p/NO_OF_FOLDS))
    print ("average recall is %f" %(r/NO_OF_FOLDS))
    print ("average f1 is %f" %(f1/NO_OF_FOLDS))

    print ("micro results are")
    print ("average precision is %f" %(p1/NO_OF_FOLDS))
    print ("average recall is %f" %(r1/NO_OF_FOLDS))
    print ("average f1 is %f" %(f11/NO_OF_FOLDS))

    
def train_LSTM(X, y, model, inp_dim, weights, epochs=EPOCHS, batch_size=BATCH_SIZE):
    cv_object = StratifiedKFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=42)
    print (cv_object)
    i = 0
    a, p, r, f1 = 0., 0., 0., 0.
    a1, p1, r1, f11 = 0., 0., 0., 0.
    sentence_len = X.shape[1]
    for train_index, test_index in cv_object.split(X,y):
        if INITIALIZE_WEIGHTS_WITH == "glove":
            model.layers[0].set_weights([weights])
        elif INITIALIZE_WEIGHTS_WITH == "random":
            shuffle_weights(model)
        else:
            print ("ERROR!")
            return
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        y_train = y_train.reshape((len(y_train), 1))
        
        
        X_temp = np.hstack((X_train, y_train))
        for epoch in range(epochs):
            for X_batch in batch_gen(X_temp, batch_size):
                x = X_batch[:, :sentence_len]
                y_temp = X_batch[:, sentence_len]

                class_weights = None
                if SCALE_LOSS_FUN:
                    class_weights = {}
                    class_weights[0] = np.where(y_temp == 0)[0].shape[0]/float(len(y_temp))
                    class_weights[1] = np.where(y_temp == 1)[0].shape[0]/float(len(y_temp))
                    class_weights[2] = np.where(y_temp == 2)[0].shape[0]/float(len(y_temp))

                try:
                    y_temp = np_utils.to_categorical(y_temp, num_classes=3)
                except Exception as e:
                    print (e)
                    #print (y_temp)
                #print (x.shape, y_temp.shape)
                loss, acc = model.train_on_batch(x, y_temp, class_weight=class_weights)
                #print (loss, acc)
        if i ==0:
            weights = model.layers[0].get_weights()[0]
        else:
            weights += model.layers[0].get_weights()[0]
        i=i+1
        print('weights.shape')
        print(weights.shape)
        y_pred = model.predict_on_batch(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        print (classification_report(y_test, y_pred))
        print (precision_recall_fscore_support(y_test, y_pred))
        #print (y_pred)
        a += accuracy_score (y_test, y_pred)
        p += precision_score(y_test, y_pred, average='weighted')
        p1 += precision_score(y_test, y_pred, average='micro')
        r += recall_score(y_test, y_pred, average='weighted')
        r1 += recall_score(y_test, y_pred, average='micro')
        f1 += f1_score(y_test, y_pred, average='weighted')
        f11 += f1_score(y_test, y_pred, average='micro')

    np.save('lstm.npy',weights/NO_OF_FOLDS)
    ff=open('vocab','wb')
    print ("macro results areeeeeeeeeeeeeeeeeeeee")
    print ("average accuracy is %f" %(a/NO_OF_FOLDS))
    print ("average precision is %f" %(p/NO_OF_FOLDS))
    print ("average recall is %f" %(r/NO_OF_FOLDS))
    print ("average f1 is %f" %(f1/NO_OF_FOLDS))

    print ("micro results are")
    print ("average precision is %f" %(p1/NO_OF_FOLDS))
    print ("average recall is %f" %(r1/NO_OF_FOLDS))
    print ("average f1 is %f" %(f11/NO_OF_FOLDS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM based models for twitter Hate speech detection')
    parser.add_argument('-f', '--embeddingfile', required=True)
    parser.add_argument('-d', '--dimension', required=True)
    parser.add_argument('--tokenizer', choices=['glove', 'nltk'], required=True)
    parser.add_argument('--loss', default=LOSS_FUN, required=True)
    parser.add_argument('--optimizer', default=OPTIMIZER, required=True)
    parser.add_argument('--epochs', default=EPOCHS, required=True)
    parser.add_argument('--batch-size', default=BATCH_SIZE, required=True)
    parser.add_argument('-s', '--seed', default=SEED)
    parser.add_argument('--folds', default=NO_OF_FOLDS)
    parser.add_argument('--kernel', default=KERNEL)
    parser.add_argument('--class_weight')
    parser.add_argument('--initialize-weights', choices=['random', 'glove'], required=True)
    parser.add_argument('--learn-embeddings', action='store_true', default=False)
    parser.add_argument('--scale-loss-function', action='store_true', default=False)


    args = parser.parse_args()
    GLOVE_MODEL_FILE = args.embeddingfile
    EMBEDDING_DIM = int(args.dimension)
    SEED = int(args.seed)
    NO_OF_FOLDS = int(args.folds)
    CLASS_WEIGHT = args.class_weight
    LOSS_FUN = args.loss
    OPTIMIZER = args.optimizer
    KERNEL = args.kernel
    if args.tokenizer == "glove":
        TOKENIZER = glove_tokenize
    elif args.tokenizer == "nltk":
        TOKENIZER = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize
    INITIALIZE_WEIGHTS_WITH = args.initialize_weights    
    LEARN_EMBEDDINGS = args.learn_embeddings
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    SCALE_LOSS_FUN = args.scale_loss_function



    np.random.seed(SEED)
    print ('GLOVE embedding: %s' %(GLOVE_MODEL_FILE))
    print ('Embedding Dimension: %d' %(EMBEDDING_DIM))
    print ('Allowing embedding learning: %s' %(str(LEARN_EMBEDDINGS)))

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_MODEL_FILE)

    tweets = select_tweets()
    #tweets = Tweets_users_hateful(tweets)
    gen_vocab()
    #filter_vocab(20000)
    X, y = gen_sequence(tweets)
    train_index, test_index = Holdout_partition(tweets)
    
    print('X.shape')
    #print(X)
    #Y = y.reshape((len(y), 1))
    MAX_SEQUENCE_LENGTH = max(map(lambda x:len(x), X))
    print ("max seq length is %d"%(MAX_SEQUENCE_LENGTH))

    data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(y)
    data, y = sklearn.utils.shuffle(data, y)
    W = get_embedding_weights()
    print('data.shape[1]')
    #print(data.shape)
    model = lstm_model(data.shape[1], EMBEDDING_DIM)
    #model = lstm_model(data.shape[1], 25, get_embedding_weights())
    #train_LSTM_Holdout(data, y, model, EMBEDDING_DIM, W,train_index, test_index, epochs=10, batch_size=128)
    train_LSTM(data, y, model, EMBEDDING_DIM, W)
    #train_LSTM_userKF(data,y, model, EMBEDDING_DIM, W)
    ff=open('vocab','wb')
    ff.write(json.dumps(vocab).encode("utf-8"))
    len(vocab)
    pdb.set_trace()
