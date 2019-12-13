import argparse
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pdb
from auxiliares import *
from models import *


def run_model_exp2():        
    #Experimento 1 Cross-Validation "Correctamente"
    tweets,_ = select_tweets('waseem',None)
    
    MAX_SEQUENCE_LENGTH = max_len(tweets)

    train_LSTM_OK(tweets, MAX_SEQUENCE_LENGTH)
    
def train_LSTM_OK(tweets, MAX_SEQUENCE_LENGTH):
    
    cv_object = KFold(n_splits = NO_OF_FOLDS, shuffle=True, random_state=42)
    a, p, r, f1 = 0., 0., 0., 0.
    a1, p1, r1, f11 = 0., 0., 0., 0.
    pn, rn, fn = 0., 0., 0.

    labels = get_labels(tweets)
    tweets, labels = sklearn.utils.shuffle(tweets,labels)
    
    for train_index, test_index in cv_object.split(tweets):
        tweets_train = []
        tweets_test = []
        for i in range(len(tweets)):
            if i in test_index:
                tweets_test.append(tweets[i])
            else:
                tweets_train.append(tweets[i])
        
        vocab = gen_vocab(tweets_train)

        X_train, y_train = gen_sequence(tweets_train, vocab, 'categorical')
        X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
    
        model = lstm_model(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,vocab)
        model.layers[0].set_weights([get_embedding_weights2(vocab)])
        
        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train))
        
        for epoch in range(EPOCHS):
            for X_batch in batch_gen(X_temp, BATCH_SIZE):
                x = X_batch[:, :MAX_SEQUENCE_LENGTH]
                y_temp = X_batch[:, MAX_SEQUENCE_LENGTH]

                class_weights = None
                
                y_temp = np_utils.to_categorical(y_temp, num_classes=3)

                loss, acc = model.train_on_batch(x, y_temp, class_weight=class_weights)

      #  shuffle_weights(model)
            
        wordEmb = model.layers[0].get_weights()[0]
       
        word2vec_model = create_model(wordEmb,vocab)
        
        X_train, y_train = gen_data(tweets_train, word2vec_model,'categorical')
        X_test, y_test = gen_data(tweets_test, word2vec_model,'categorical')
        
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