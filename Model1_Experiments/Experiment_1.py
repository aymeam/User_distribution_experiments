import argparse
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pdb
from auxiliares import *

def run_model_exp1(flag, strategy):        
    #Experimento 1 Cross-Validation
    tweets,_ = select_tweets('waseem',strategy)
    vocab = gen_vocab(tweets)
    X, y = gen_sequence(tweets, vocab, flag)
    #Y = y.reshape((len(y), 1))
    MAX_SEQUENCE_LENGTH = max(map(lambda x:len(x), X))
    print ("max seq length is %d"%(MAX_SEQUENCE_LENGTH))
    data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(y)
    data, y = sklearn.utils.shuffle(data, y)
    W = get_embedding_weights()
    print(data.shape)
    if flag == 'binary':
        model = lstm_model_bin(data.shape[1], EMBEDDING_DIM)
    else:
        model = lstm_model(data.shape[1], EMBEDDING_DIM)
    train_LSTM_variante1(tweets,data, y, model, EMBEDDING_DIM, W, 10, 128,flag)

def train_LSTM_variante1(tweets,X, y, model, inp_dim, weights, epochs, batch_size,flag):
    #Step 1: Training the embeddings with all data
    a, p, r, f1 = 0., 0., 0., 0.
    a1, p1, r1, f11 = 0., 0., 0., 0.
    pn, rn, fn = 0., 0., 0.
    sentence_len = X.shape[1]
    current_fold=0
    y_train = y.reshape((len(y), 1))
    X_temp = np.hstack((X, y_train))
    for epoch in range(epochs):
        for X_batch in batch_gen(X_temp, batch_size):
            x = X_batch[:, :sentence_len]
            y_temp = X_batch[:, sentence_len]
            try:
                if flag != 'binary':
                    y_temp = np_utils.to_categorical(y_temp, num_classes=3)
            except Exception as e:
                print (e)
            loss, acc = model.train_on_batch(x, y_temp, class_weight=None)
    #Extracting learned embeddings 
    wordEmb = model.layers[0].get_weights()[0]
  
    #Step 2: Cross- Validation Using the XGB classifier and the learned embeddings
    tweets, word2vec_model =select_tweets_whose_embedding_exists('waseem',wordEmb)
    X, y = gen_data(tweets, word2vec_model,flag)
    cv_object = StratifiedKFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=42)
    for train_index, test_index in cv_object.split(X,y):
        tweets_train = []
        tweets_test = []
        for i in range(len(tweets)):
            if i in test_index:
                tweets_test.append(tweets[i])
            else:
                tweets_train.append(tweets[i])
                
        X_train, y_train = gen_data(tweets_train,word2vec_model,flag)
        X_test, y_test = gen_data(tweets_test,word2vec_model,flag)

        precision, recall, f1_score, acc, p_weighted, p_macro, r_weighted, r1_macro, f1_weighted, f11_macro = gradient_boosting_classifier(tweets, wordEmb,[],[], X_train, y_train, X_test, y_test,flag)
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
    parser = argparse.ArgumentParser(description='LSTM based models for twitter Hate speech detection')
    parser.add_argument('-t', '--type',choices=['binary', 'categorical'], default = 'categorical')
    TOKENIZER = 'glove'
    GLOVE_MODEL_FILE = 'glove.txt'
    EMBEDDING_DIM = 200
    OPTIMIZER = 'adam'
    INITIALIZE_WEIGHTS_WITH = 'random'
    LEARN_EMBEDDINGS = True
    EPOCHS = 10
    BATCH_SIZE = 128
    SCALE_LOSS_FUN = None
    SEED = 42
    np.random.seed(SEED)
    print ('GLOVE embedding: %s' %(GLOVE_MODEL_FILE))
    print ('Embedding Dimension: %d' %(EMBEDDING_DIM))
    print ('Allowing embedding learning: %s' %(str(LEARN_EMBEDDINGS)))
    flag = parser.parse_args().type
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_MODEL_FILE)
    run_model_exp1(flag, None)
