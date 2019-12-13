import argparse
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pdb
from auxiliares import *
from models import *

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
   
    model = lstm_model(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,vocab)
    
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
    
    tweets = select_tweets_whose_embedding_exists(tweets, word2vec_model)

    X, y = gen_data(tweets, word2vec_model,'categorical')

    cv_object = StratifiedKFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=42)

    for train_index, test_index in cv_object.split(X,y):
        X_train, y_train = X[train_index],y[train_index]
        X_test, y_test = X[test_index],y[test_index]
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)

        X_test, y_test = sklearn.utils.shuffle(X_test, y_test)
        model = gradient_boosting_classifier(X_train, y_train)
        precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem =evaluate_model(model, X_test, y_test, 'categorical')
        
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
