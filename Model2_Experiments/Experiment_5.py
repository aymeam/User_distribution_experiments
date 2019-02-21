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

def run_model_exp5(oversampling_rate, vector_type, embed_size,flag):   
    #sin usuarios solapados en training y testin
    if flag == 'binary':
        model_type = "binary_blstm"
    else:
        model_type = "blstm"
    tweets = pickle.load(open('DatosCSV/data_sorted.pkl', 'rb'))
    x_text, labels = load_data('data_sorted')
    

    #cv_object = KFold(n_splits=10, shuffle=False, random_state=42)
    a, p, r, f1 = 0., 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    pn, rn, fn = 0., 0., 0.
    test_indexes= cv_sorted_data(x_text)
    all_names = []

    for test_index in test_indexes:
        X_train, y_train, X_test, y_test,names_train,names_test = [],[],[],[],[],[]
        for i in range(len(x_text)):
            if i  not in test_index:
                X_train.append(x_text[i])
                y_train.append(labels[i])
                names_train.append(tweets[i]['name'])
                all_names.append(tweets[i]['name'])
            else:
                X_test.append(x_text[i])
                y_test.append(labels[i])
                names_test.append(tweets[i]['name'])
                all_names.append(tweets[i]['name'])

        #verificando solapamiento de usuarios 
        print(all_names[3497],all_names[3498],all_names[3499],all_names[3500],all_names[3501])
        for name in names_train:
            if name in names_test:
                print('usuario solapado: ' + name)
                print(names_train[0],names_train[len(names_train)-1], names_test[0], names_test[len(names_test)-1])       
        data_dict = data_processor(x_text,X_train,y_train,X_test,y_test,flag)
        X_train,y_train = oversampling(X_train,y_train,oversampling_rate)

        precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem = train(data_dict, model_type, vector_type,flag, embed_size)    

        p += precisionw
        p1 += precisionm
        r += recallw
        r1 += recallm
        f1 += f1_scorew
        f11 += f1_scorem
        pn += precision
        rn += recall
        fn += f1_score
    NO_OF_FOLDS = 10
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

if __name__ == "__main__":
    vector_type = "sswe"
    oversampling_rate = 3
    NO_OF_FOLDS = 5
    embed_size = 50
    flag = 'binary'
    run_model_exp5(oversampling_rate,  vector_type, embed_size,flag)