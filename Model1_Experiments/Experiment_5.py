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
    tweets,_ = select_tweets('sorted',strategy)
    vocab = gen_vocab(tweets)
    X, y = gen_sequence(tweets, vocab, flag)
    #Y = y.reshape((len(y), 1))
    MAX_SEQUENCE_LENGTH = max(map(lambda x:len(x), X))
    print ("max seq length is %d"%(MAX_SEQUENCE_LENGTH))
    data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(y)
    data, y = sklearn.utils.shuffle(data, y)
    W = get_embedding_weights()
    print(data.shape)
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
    print(len(test_indexes))
    for test_index in test_indexes:
        if INITIALIZE_WEIGHTS_WITH == "glove":
            model.layers[0].set_weights([weights])
        elif INITIALIZE_WEIGHTS_WITH == "random":
            shuffle_weights(model)
        else:
            print ("ERROR!")
            return
        print('SHAPES')
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

                try:
                    y_temp = y_temp#np_utils.to_categorical(y_temp, num_classes=3)
                except Exception as e:
                    print (e)
                #print (x.shape, y_temp.shape)
                loss, acc = model.train_on_batch(x, y_temp, class_weight=class_weights)
                #print (loss, acc)
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
            print(tweets[i]['label'])
            if i in test_index:
                tweets_test.append(tweets[i])
            else:
                tweets_train.append(tweets[i])
                print('siiiiiiiiiiiiii')
        X_train, y_train = gen_data(tweets_train,word2vec_model,flag)
        X_test, y_test = gen_data(tweets_test,word2vec_model,flag)
        print(y_test)
    
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
    print ("None average results are:")
    print (prod(pn, NO_OF_FOLDS))
    print (prod(rn, NO_OF_FOLDS))
    print (prod(fn, NO_OF_FOLDS))       
    
    print ("weighted results are")
    print ("average accuracy is %f" %(a/NO_OF_FOLDS))
    print ("average precision is %f" %(p/NO_OF_FOLDS))
    print ("average recall is %f" %(r/NO_OF_FOLDS))
    print ("average f1 is %f" %(f1/NO_OF_FOLDS))

    print ("macro results are")
    print ("average precision is %f" %(p1/NO_OF_FOLDS))
    print ("average recall is %f" %(r1/NO_OF_FOLDS))
    print ("average f1 is %f" %(f11/NO_OF_FOLDS))

if __name__ == "__main__":
    vector_type = "sswe"
    oversampling_rate = 3
    NO_OF_FOLDS = 10
    embed_size = 50
    run_model_exp5('binary',None)
