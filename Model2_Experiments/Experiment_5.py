import argparse
import numpy as np
import sklearn.utils
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
    model_type = "binary_blstm"
    tweets = pickle.load(open('../Data/Data_new.pkl', 'rb'))
    x_text, labels = load_data('data_new')
    

    a, p, r, f1 = 0., 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    pn, rn, fn = 0., 0., 0.
    test_indexes= cv_sorted_data(x_text)
    parts  = 0
    
    for test_index in test_indexes:
        print('parts')
        print(parts)
        parts += 1
        x_text, labels = sklearn.utils.shuffle(x_text, labels)
        X_train, y_train, X_test, y_test,names_train,names_test = [],[],[],[],[],[]
        for i in range(len(x_text)):
            if i  not in test_index:
                X_train.append(x_text[i])
                y_train.append(labels[i])
                names_train.append(tweets[i]['name'])
            else:
                X_test.append(x_text[i])
                y_test.append(labels[i])
                names_test.append(tweets[i]['name'])
      
        #verificando solapamiento de usuarios 
#         solapados = 0
#         cont =0
#         for name in names_test:
#             if name in names_train:
#                 print('usuario solapado' + name + ' ' + str(cont))
#                 solapados += 1
#             cont +=1
        
#         print(str(solapados) + ' usuarios solapados')  

        data_dict = data_processor(x_text,X_train,y_train,X_test,y_test,flag)


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
    print_scores(p, p1, r,r1, f1, f11,pn, rn, fn,NO_OF_FOLDS)


if __name__ == "__main__":
    vector_type = "sswe"
    oversampling_rate = 
    embed_size = 50
    flag = 'binary'
    run_model_exp5(oversampling_rate,  vector_type, embed_size,flag)