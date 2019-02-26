import argparse
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
import os
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix 
from tensorflow.contrib import learn
import tflearn
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from auxiliares import *

def run_model_exp5(flag,strategy):        
    #Experimento 1 Cross-Validation
    tweets,_ = select_tweets('sem_eval',strategy)
    vocab = gen_vocab(tweets)
    X, y = gen_sequence(tweets, vocab, flag)
    #Y = y.reshape((len(y), 1))
    MAX_SEQUENCE_LENGTH = max(map(lambda x:len(x), X))
    print ("max seq length is %d"%(MAX_SEQUENCE_LENGTH))
    data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(y)
    data, y = sklearn.utils.shuffle(data, y)
    W = get_embedding_weights()
    model = lstm_model_bin(data.shape[1], EMBEDDING_DIM)
    run_model_CV_wellDistributed(tweets,data, y, model, EMBEDDING_DIM, W, 10, 128,flag)

def run_model_CV_wellDistributed(tweets,X, y, model, inp_dim, weights,epochs,batch_size,flag):
    cv_object = StratifiedKFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=42)
    print (cv_object)
    a, p, r, f1 = 0., 0., 0., 0.
    a1, p1, r1, f11 = 0., 0., 0., 0.
    pn, rn, fn = 0., 0., 0.
    test_indexes= cv_sorted_data(X)

    sentence_len = X.shape[1]
    print('run_model_CV_wellDistributed')
    for test_index in test_indexes:
        if INITIALIZE_WEIGHTS_WITH == "glove":
            model.layers[0].set_weights([weights])
        elif INITIALIZE_WEIGHTS_WITH == "random":
            shuffle_weights(model)
        else:
            print ("ERROR!")
            return
        X_train, y_train, X_test, y_test = [],[],[],[]
        for i in range(len(X)):
            if i in test_index:
                X_train.append(X[i])
                y_train.append(y[i])
            else:
                X_test.append(X[i])
                y_test.append(y[i])
                
        y_train = np.array(y_train)
        X_train = np.array(X_train)
        y_test = np.array(y_test)
        X_test = np.array(X_test)
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

                loss, acc = model.train_on_batch(x, y_temp, class_weight=class_weights)
         
        temp = model.predict_on_batch(X_test)
        y_pred = []
        for i in temp:
            if i[0] >0.5:
                y_pred.append(1)
            else:
                y_pred.append(0) 
        print (classification_report(y_test, y_pred))
        wordEmb = model.layers[0].get_weights()[0]
        
        tweets, word2vec_model =select_tweets_whose_embedding_exists('sorted',wordEmb)
        tweets_train = []
        tweets_test = []
        for i in range(len(tweets)):
            if i in test_index:
                tweets_test.append(tweets[i])
            else:
                tweets_train.append(tweets[i])
        X_train, y_train = gen_data(tweets_train,word2vec_model,flag)
        X_test, y_test = gen_data(tweets_test,word2vec_model,flag)
   
        precision, recall, f1_score, acc, p_weighted, p_macro, r_weighted, r1_macro, f1_weighted, f11_macro = gradient_boosting_classifier([], wordEmb,[],[], X_train, y_train, X_test, y_test,flag)
        a += acc
        p += p_weighted
        p1 += p_macro
        r += r_weighted
        r1 += r1_macro
        f1 += f1_weighted
        f11 += f11_macro
        pn += precision
        rn += recall
        fn += f1_score
    print_scores(p, p1, r,r1, f1, f11,pn, rn, fn,NO_OF_FOLDS)

if __name__ == "__main__":
    vector_type = "sswe"
    oversampling_rate = 3
    NO_OF_FOLDS = 10
    embed_size = 50
    run_model_exp5('binary',None)
