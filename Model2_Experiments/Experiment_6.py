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

def run_model_exp6(oversampling_rate, vector_type, embed_size,flag): 
    model_type = "binary_blstm"
    X_train, y_train = load_data('data_new')
    X_train, y_train = oversampling(X_train, y_train,oversampling_rate)
 
    X_test, y_test = load_data('test')
    
    x_text = np.concatenate((X_train, X_test), axis=0)

    data_dict = data_processor(x_text,X_train,y_train,X_test,y_test,flag)
    a, p, r, f1 = 0., 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    pn, rn, fn = 0., 0., 0.
    precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem = train(data_dict, model_type, vector_type,flag, embed_size)    

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
    
    print_scores(p, p1, r,r1, f1, f11,pn, rn, fn,1)

    
if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="Experiment 3. Results using the Waseem & Hovy’s dataset as training set and SemEval 2019 as testing set")
#    parser.add_argument('--type', choices=['binary', 'categorical'], default = 'categorical')
    vector_type = "sswe"
    oversampling_rate = 3
    #flag=parser.parse_args().type
    NO_OF_FOLDS = 5
    embed_size = 50
    flag = 'binary'
    run_model_exp6(oversampling_rate,  vector_type, embed_size,flag)