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

def run_model_exp3(oversampling_rate, vector_type, embed_size,flag): 
    model_type = "blstm"
    X_train, y_train = load_data('train')
    X_train, y_train = oversampling(X_train, y_train,3)
 
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
    
if __name__ == "__main__":
    vector_type = "sswe"
    oversampling_rate = 3
    NO_OF_FOLDS = 5
    embed_size = 50
    flag = 'cross_domain_waseem'
    run_model_exp3(oversampling_rate,  vector_type, embed_size,flag)