import argparse
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pdb
from auxiliares import *

def run_model_exp1():        
    #Experimento 1 Cross-Validation
    tweets,_ = select_tweets('waseem',None)
    
    vocab = gen_vocab(tweets)
    
    MAX_SEQUENCE_LENGTH = max_len(tweets)
    
    train_LSTM_variante1(tweets,vocab, MAX_SEQUENCE_LENGTH)

def train_LSTM_variante1(tweets, vocab, MAX_SEQUENCE_LENGTH):
    #Step 1: Training the embeddings with all data
    a, p, r, f1 = 0., 0., 0., 0.
    a1, p1, r1, f11 = 0., 0., 0., 0.
    pn, rn, fn = 0., 0., 0.

    X, y = gen_sequence(tweets, vocab, 'categorical' )
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    X, y = sklearn.utils.shuffle(X, y)
    y = np.array(y)
    
    y_train = y.reshape((len(y), 1))
    X_temp = np.hstack((X, y_train))
   
    model = lstm_model(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
    #initializing with random embeddings
    shuffle_weights(model)
    
    for epoch in range(EPOCHS):
        for X_batch in batch_gen(X_temp, BATCH_SIZE):
            x = X_batch[:, :MAX_SEQUENCE_LENGTH]
            y_temp = X_batch[:, MAX_SEQUENCE_LENGTH]
            try:
                y_temp = np_utils.to_categorical(y_temp, num_classes=3)
            except Exception as e:
                print (e)
            loss, acc = model.train_on_batch(x, y_temp, class_weight=None)

  
    tweets,_ = select_tweets('waseem', None)
        
    #Extracting learned embeddings 
    wordEmb = model.layers[0].get_weights()[0]
  
    #Step 2: Cross- Validation Using the XGB classifier and the learned embeddings
    word2vec_model = create_model(wordEmb,vocab)

    X, y = gen_data(tweets, word2vec_model,'categorical')
    cv_object = KFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=42)
    
   

    for train_index, test_index in cv_object.split(X):
        X_train, y_train = X[train_index],y[train_index]
        X_test, y_test = X[test_index],y[test_index]
        
        precision, recall, f1_score, acc, p_weighted, p_macro, r_weighted, r1_macro, f1_weighted, f11_macro = gradient_boosting_classifier(X_train, y_train,X_test, y_test,'categorical')
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
        
    print_scores(p, p1, r,r1, f1, f11,pn, rn, fn, NO_OF_FOLDS)    
    
if __name__ == "__main__":
    TOKENIZER = 'glove'
    GLOVE_MODEL_FILE = 'glove.txt'
    EMBEDDING_DIM = 200
    OPTIMIZER = 'adam'
    LEARN_EMBEDDINGS = True
    EPOCHS = 10
    BATCH_SIZE = 128
    SCALE_LOSS_FUN = None
    NO_OF_FOLDS = 10
    SEED = 42
    np.random.seed(SEED)
    run_model_exp1()
