import argparse
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pdb
from auxiliares import *


def run_exp_4(flag, strategy):
    #Experimento3 Holdhout partition
    tweets,train_users = select_tweets('waseem',strategy)

    vocab = gen_vocab(tweets)

    MAX_SEQUENCE_LENGTH = max_len(tweets)

    train_index, test_index = Holdout_partition(tweets,train_users)

    train_LSTM_Holdout(tweets, train_index, test_index, vocab, MAX_SEQUENCE_LENGTH)


def train_LSTM_Holdout(tweets, train_index, test_index, vocab, MAX_SEQUENCE_LENGTH):
    
    X, y = gen_sequence(tweets,vocab,flag)

    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y= np.array(y)
    
    model = lstm_model(X.shape[1], EMBEDDING_DIM)

    tweets_train = []
    tweets_test = []
    for i in range(len(tweets)):
        if i in train_index:
            tweets_train.append(tweets[i])
        elif i in test_index:
            tweets_test.append(tweets[i])
            
    X_train, y_train = gen_sequence(tweets_train, vocab, 'categorical')
    X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test, y_test = gen_sequence(tweets_test, vocab, 'binary')
    X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    y_train = y_train.reshape((len(y_train), 1))
    X_temp = np.hstack((X_train, y_train))
    sentence_len = X.shape[1]
    shuffle_weights(model)
    
    for epoch in range(EPOCHS):
        for X_batch in batch_gen(X_temp, BATCH_SIZE):
            x = X_batch[:, :sentence_len]
            y_temp = X_batch[:, sentence_len]

            y_temp = np_utils.to_categorical(y_temp, num_classes=3)

            loss, acc = model.train_on_batch(x, y_temp, class_weight=None)
    wordEmb = model.layers[0].get_weights()[0]
    
    word2vec_model = create_model(wordEmb,vocab)

    
    X_train, y_train = gen_data(tweets_train,word2vec_model,flag)
    X_test, y_test = gen_data(tweets_test,word2vec_model,flag)
    
    precision, recall, f1_score, acc, p_weighted, p_macro, r_weighted, r1_macro, f1_weighted, f11_macro = gradient_boosting_classifier(X_train, y_train,X_test, y_test, 'cross_waseem')

    print_scores(p_weighted, p_macro, r_weighted,r1_macro, f1_weighted, f11_macro,precision, recall, f1_score, 1)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM based models for twitter Hate speech detection')
    parser.add_argument('-t', '--type',choices=['binary', 'categorical'], default = 'binary')
    
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
    flag = parser.parse_args().type
    strategy = 1
    run_exp_4(flag, strategy)