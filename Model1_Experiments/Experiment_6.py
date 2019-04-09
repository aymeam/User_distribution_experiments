import argparse
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pdb
from auxiliares import *


def run_exp_6 (flag, strategy):
    #Experimento Cross Domain dataset partition
    tweets_train,_ = select_tweets('data_new',strategy)
    tweets_test,_ =  select_tweets('sem_eval',strategy)
    
    vocab = gen_vocab(tweets_train)

    MAX_SEQUENCE_LENGTH = max_len(tweets_train)
    
    train_LSTM_Cross_Domain(tweets_train,tweets_test,MAX_SEQUENCE_LENGTH)   
    
def train_LSTM_Cross_Domain(tweets_train,tweets_test,MAX_SEQUENCE_LENGTH):
        a, p, r, f1 = 0., 0., 0., 0.
        a1, p1, r1, f11 = 0., 0., 0., 0.
        pn,rn,fn = 0.,0.,0.
        
        model = lstm_model_bin(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

        shuffle_weights(model)
        
        X_train, y_train = gen_sequence(tweets_train,vocab,'binary')
        
        X_test, y_test = gen_sequence(tweets_test,vocab,'binary')
        
        X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
        X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
        
        y_train = np.array(y_train)
        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train))
        
        for epoch in range(EPOCHS):
            for X_batch in batch_gen(X_temp, BATCH_SIZE):
                x = X_batch[:, :MAX_SEQUENCE_LENGTH]
                y_temp = X_batch[:, MAX_SEQUENCE_LENGTH]

                class_weights = None
                loss, acc = model.train_on_batch(x, y_temp, class_weight=class_weights)
                #print (loss, acc)
                
        temp = model.predict_on_batch(X_test)
        y_pred=[]
        for i in temp:
            if i[0] > 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
#         print (classification_report(y_test, y_pred))
#         print (precision_recall_fscore_support(y_test, y_pred))
        
        wordEmb = model.layers[0].get_weights()[0]
        
        word2vec_model = create_model(wordEmb,vocab)

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
        print_scores(p, p1, r,r1, f1, f11,pn, rn, fn, 1)

    
if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='LSTM based models for twitter Hate speech detection')
    #parser.add_argument('-t', '--type',choices=['binary', 'categorical'], default = 'categorical')
    TOKENIZER = 'glove'
    GLOVE_MODEL_FILE = 'glove.txt'
    EMBEDDING_DIM = 200
    OPTIMIZER = 'adam'
    INITIALIZE_WEIGHTS_WITH = 'glove'
    LEARN_EMBEDDINGS = True
    EPOCHS = 10
    BATCH_SIZE = 128
    SCALE_LOSS_FUN = None
    SEED = 42
    np.random.seed(SEED)

    run_exp_6(flag, None)
