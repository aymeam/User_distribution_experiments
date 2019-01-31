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


#Réplica del experimento original descrito en la literatura
def run_model_exp1(oversampling_rate, vector_type, embed_size,flag):    
    if flag == 'binary':
        model_type = "binary_blstm"
    else:
        model_type = "blstm"
    
    #Oversampling before cross-validation
    x_text, labels = get_data(oversampling_rate,'train',flag)
    
    #cross-validation with oversampled data
    cv_object = KFold(n_splits=5, shuffle=True, random_state=42)
    a, p, r, f1 = 0., 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    pn, rn, fn = 0., 0., 0.
    NO_OF_FOLDS = 5

    for train_index, test_index in cv_object.split(x_text):
        X_train, y_train, X_test, y_test = [],[],[],[]
        for i in range(len(x_text)):
            if i in train_index:
                X_train.append(x_text[i])
                y_train.append(labels[i])
            else:
                X_test.append(x_text[i])
                y_test.append(labels[i])

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
    print_scores(p, p1, r,r1, f1, f11,pn, rn, fn,NO_OF_FOLDS)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment 0. Original Experiment Replica')
    parser.add_argument('--type', choices=['binary', 'categorical'], default = 'categorical')
    vector_type = "sswe"
    oversampling_rate = 3
    flag=parser.parse_args().type
    NO_OF_FOLDS = 5
    embed_size = 50
    run_model_exp1(oversampling_rate,  vector_type, embed_size,flag)
