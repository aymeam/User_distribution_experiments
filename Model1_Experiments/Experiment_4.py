import argparse
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pdb
from auxiliares import *


def run_exp_4(flag, strategy):
    #Experimento3 Holdhout partition
    tweets,train_users = select_tweets('waseem',strategy)
    print(train_users)
    print('run_experiment3')
    print(len(tweets))
    vocab = gen_vocab(tweets)

    X, y = gen_sequence(tweets,vocab,flag)
    train_index, test_index = Holdout_partition(tweets,train_users)

    MAX_SEQUENCE_LENGTH = max(map(lambda x:len(x), X))

    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y= np.array(y)
    sentence_len = X.shape[1]
    
    W = get_embedding_weights()
    if flag == 'binary':
        model = lstm_model_bin(X.shape[1], EMBEDDING_DIM)
    else:
        model = lstm_model(X.shape[1], EMBEDDING_DIM)    #model = lstm_model(X.shape[1], 25, get_embedding_weights())
    train_LSTM_Holdout(tweets,X,y, model, EMBEDDING_DIM, W, train_index, test_index,flag, epochs=EPOCHS, batch_size=BATCH_SIZE)


def train_LSTM_Holdout(tweets,X,y, model, inp_dim, weights, train_index, test_index,flag, epochs, batch_size):
    X_train, y_train = X[train_index], y[train_index]
    tweets_train = []
    tweets_test = []
    for i in range(len(tweets)):
        if i in train_index:
            tweets_train.append(tweets[i])
        elif i in test_index:
            tweets_test.append(tweets[i])
            
    print('train_LSTM_Holdout')
    if flag != 'binary':
        X_test, y_test = X[test_index], y[test_index]
    else:
        X_test, y_test_aux = X[test_index], y[test_index]

    print("Counter train")
    from collections import Counter
    print(Counter(y_train))
    
    print('train_LSTM_Holdout')
    print(len(tweets))
    print(len(train_index))
    print(len(test_index))
    y_train = y_train.reshape((len(y_train), 1))
    X_temp = np.hstack((X_train, y_train))
    sentence_len = X.shape[1]

    for epoch in range(epochs):
        for X_batch in batch_gen(X_temp, batch_size):
            x = X_batch[:, :sentence_len]
            y_temp = X_batch[:, sentence_len]

            class_weights = None
            if SCALE_LOSS_FUN:
                class_weights = {}
                class_weights[0] = np.where(y_temp == 0)[0].shape[0]/float(len(y_temp))
                class_weights[1] = np.where(y_temp == 1)[0].shape[0]/float(len(y_temp))
                class_weights[2] = np.where(y_temp == 2)[0].shape[0]/float(len(y_temp))

            try:
                if flag != 'binary':
                    y_temp = np_utils.to_categorical(y_temp, num_classes=3)
            except Exception as e:
                print (e)
                print (y_temp)
            loss, acc = model.train_on_batch(x, y_temp, class_weight=class_weights)
    temp = model.predict_on_batch(X_test)
    if flag != 'binary': 
        y_pred = np.argmax(temp, axis=1)
        
        #y_test = np.argmax(y_test, axis=1)
        print(y_pred[0])
        print(y_test[0])
    else:
        y_pred = []
        for i in temp:
            if i[0] >0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)  
        
        print(y_test_aux[0:50])
        y_test = []
        print(y_test[0:100])
        for i in y_test_aux:
            if i >0.5:
                y_test.append(1)
            else:
                y_test.append(0) 

    print (classification_report(y_test, y_pred))
    print (precision_recall_fscore_support(y_test, y_pred))

    wordEmb = model.layers[0].get_weights()[0]
    X_train, y_train = gen_data(tweets_train,word2vec_model,flag)
    X_test, y_test = gen_data(tweets_test,word2vec_model,flag)
    
    precision, recall, f1_score, acc, p_weighted, p_macro, r_weighted, r1_macro, f1_weighted, f11_macro = gradient_boosting_classifier([], wordEmb,[],[], X_train, y_train, X_test, y_test,flag)

    print_scores(p_weighted, p_macro, r_weighted,r1_macro, f1_weighted, f11_macro,precision, recall, f1_score, 1)

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
    strategy = 2
    run_exp_4(flag, strategy)