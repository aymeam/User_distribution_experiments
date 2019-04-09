import argparse
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pdb
from auxiliares import *


def run_model_exp2():        
    #Experimento 1 Cross-Validation "Correctamente"
    tweets,_ = select_tweets('waseem',None)
    
    vocab = gen_vocab(tweets)

    MAX_SEQUENCE_LENGTH = max_len(tweets)

    train_LSTM_OK(tweets, vocab, MAX_SEQUENCE_LENGTH)
    
def train_LSTM_OK(tweets,vocab, MAX_SEQUENCE_LENGTH):
    
    cv_object = KFold(n_splits = NO_OF_FOLDS, shuffle=True, random_state=42)
    a, p, r, f1 = 0., 0., 0., 0.
    a1, p1, r1, f11 = 0., 0., 0., 0.
    pn, rn, fn = 0., 0., 0.
    
    model = lstm_model(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

    for train_index, test_index in cv_object.split(tweets):
        tweets_train = []
        tweets_test = []
        for i in range(len(tweets)):
            if i in test_index:
                tweets_test.append(tweets[i])
            else:
                tweets_train.append(tweets[i])
        
        shuffle_weights(model)
            
        X_train, y_train = gen_sequence(tweets_train, vocab, 'categorical')
        X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        X_test, y_test = gen_sequence(tweets_test, vocab, 'categorical')
        X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        
        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train))
        
        for epoch in range(EPOCHS):
            for X_batch in batch_gen(X_temp, BATCH_SIZE):
                x = X_batch[:, :MAX_SEQUENCE_LENGTH]
                y_temp = X_batch[:, MAX_SEQUENCE_LENGTH]

                class_weights = None
                
                y_temp = np_utils.to_categorical(y_temp, num_classes=3)

                loss, acc = model.train_on_batch(x, y_temp, class_weight=class_weights)
                
                
        wordEmb = model.layers[0].get_weights()[0]
       
        word2vec_model = create_model(wordEmb,vocab)
        
#         tweets_train = select_tweets_whose_embedding_exists(tweets_train, word2vec_model)
#         tweets_test = select_tweets_whose_embedding_exists(tweets_test, word2vec_model)

        
        X_train, y_train = gen_data(tweets_train, word2vec_model,'categorical')
        X_test, y_test = gen_data(tweets_test,word2vec_model,'categorical')
        
        precision, recall, f1_score, acc, p_weighted, p_macro, r_weighted, r1_macro, f1_weighted, f11_macro = gradient_boosting_classifier(X_train, y_train, X_test, y_test, 'categorical')

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
    TOKENIZER = 'glove'
    GLOVE_MODEL_FILE = 'glove.txt'
    EMBEDDING_DIM = 200
    OPTIMIZER = 'adam'
    INITIALIZE_WEIGHTS_WITH = 'random'
    LEARN_EMBEDDINGS = True
    EPOCHS = 10
    NO_OF_FOLDS = 10
    BATCH_SIZE = 128
    SCALE_LOSS_FUN = None
    SEED = 42
    np.random.seed(SEED)

    run_model_exp2()