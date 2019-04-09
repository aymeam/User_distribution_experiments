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
print('NO_OF_FOLDS')
print(NO_OF_FOLDS)
def run_model_exp5(flag,strategy):        
    #Experimento 1 Cross-Validation
    tweets,_ = select_tweets('data_new',strategy)
    
    vocab = gen_vocab(tweets)
    
    MAX_SEQUENCE_LENGTH = max_len(tweets)
    
    run_model_CV_wellDistributed(tweets,MAX_SEQUENCE_LENGTH,vocab, flag)

def run_model_CV_wellDistributed(tweets,MAX_SEQUENCE_LENGTH,vocab, flag):
    a, p, r, f1 = 0., 0., 0., 0.
    a1, p1, r1, f11 = 0., 0., 0., 0.
    pn, rn, fn = [0,0], [0,0], [0,0]
    
    test_indexes= cv_sorted_data(tweets)
    X, y = gen_sequence(tweets, vocab, 'binary')

    sentence_len = MAX_SEQUENCE_LENGTH
    model = lstm_model_bin(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
    tweets, y = sklearn.utils.shuffle(tweets, y)
    
    for test_index in test_indexes:
        weights = get_embedding_weights(vocab)
        model.layers[0].set_weights([weights])
#        shuffle_weights(model)
 
        names_train = []
        names_test = []
        tweets_train = []
        tweets_test = []
        for i in range(len(tweets)):
            if i in test_index:
                tweets_test.append(tweets[i])
            else:
                tweets_train.append(tweets[i])
                
        X_train, y_train = gen_sequence(tweets_train, vocab, 'binary')
        
        X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
        

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_test, y_test = gen_sequence(tweets_test, vocab, 'binary')
        X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        solapados = 0
        cont =0
        for name in names_test:
            if name in names_train:
                print('usuario solapado' + name + ' ' + str(cont))
                solapados += 1
            cont +=1
        print(str(solapados) + ' usuarios solapados')        

        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train))
        
        for epoch in range(EPOCHS):
            for X_batch in batch_gen(X_temp, BATCH_SIZE):
                x = X_batch[:, :sentence_len]
                y_temp = X_batch[:, sentence_len]

                class_weights = None
                loss, acc = model.train_on_batch(x, y_temp, class_weight=class_weights)
         
        temp = model.predict_on_batch(X_test)
#         y_pred = []
#         for i in temp:
#             if i[0] >0.5:
#                 y_pred.append(1)
#             else:
#                 y_pred.append(0) 
#         print (classification_report(y_test, y_pred))
        
        wordEmb = model.layers[0].get_weights()[0]
        
        word2vec_model = create_model(wordEmb,vocab)
        
        tweets_train =select_tweets_whose_embedding_exists(tweets_train,word2vec_model)
        tweets_test =select_tweets_whose_embedding_exists(tweets_test,word2vec_model)
        
        
 
        X_train, y_train = gen_data(tweets_train,word2vec_model,flag)

        X_test, y_test = gen_data(tweets_test,word2vec_model,flag)

        precision, recall, f1_score, acc, p_weighted, p_macro, r_weighted, r1_macro, f1_weighted, f11_macro = gradient_boosting_classifier(X_train, y_train, X_test, y_test, 'binary')
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
        print('rn')
        print(rn)

    print_scores(p, p1, r,r1, f1, f11,pn, rn, fn,NO_OF_FOLDS)

if __name__ == "__main__":
    oversampling_rate = 3
    TOKENIZER = 'glove'
    EMBEDDING_DIM = 200
    OPTIMIZER = 'adam'
    LEARN_EMBEDDINGS = True
    EPOCHS = 10
    BATCH_SIZE = 128
    SCALE_LOSS_FUN = None
    SEED = 42
    embed_size = 50
    run_model_exp5('binary',None)