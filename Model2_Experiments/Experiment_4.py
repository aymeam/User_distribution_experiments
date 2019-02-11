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

def run_model_exp4(oversampling_rate, vector_type, embed_size,strategy,flag):    

    #data_dict = Holdout_partition(data,1)
    if flag == 'binary':
        model_type = 'binary_blstm'
        data_dict = Holdout_partition(oversampling_rate,strategy,flag)
    else:
        model_type = 'blstm'
        data_dict = Holdout_partition(oversampling_rate,strategy,flag)

    a, p, r, f1 = 0., 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    pn, rn, fn = 0., 0., 0.
    
    
    precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem = train(data_dict, model_type, vector_type,flag, embed_size)    
  
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Results using partitions of the Waseem & Hovy’sdataset into train set and test set considering the userdistribution (no overlapping users between train and test sets')
    parser.add_argument('--type', choices=['binary', 'categorical'], default = 'categorical')
    vector_type = "sswe"
    oversampling_rate = 3
    flag='binary'#parser.parse_args().type
    NO_OF_FOLDS = 5
    embed_size = 50
    strategy = 3
    run_model_exp4(oversampling_rate, vector_type, embed_size,strategy,flag)
