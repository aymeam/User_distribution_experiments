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
    print(len(tweets_train))
    print(len(tweets_test))
    X_train, y_train = gen_sequence(tweets_train,vocab,'binary')
    X_test, y_test = gen_sequence(tweets_test,vocab,'binary')
    
    W = get_embedding_weights()
    MAX_SEQUENCE_LENGTH = max(map(lambda x:len(x), X_train))

    data = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
    data_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
    
    model = lstm_model_bin(data.shape[1], EMBEDDING_DIM)
    train_LSTM_Cross_Domain(tweets_train,tweets_test,data, y_train, data_test, y_test, model, EMBEDDING_DIM, W, 10)   
    
def train_LSTM_Cross_Domain(tweets_train,tweets_test,X_train, y_train, X_test, y_test, model, EMBEDDING_DIM, W, epochs):
        a, p, r, f1 = 0., 0., 0., 0.
        a1, p1, r1, f11 = 0., 0., 0., 0.
        pn,rn,fn = 0.,0.,0.
        NO_OF_FOLDS = 1
        sentence_len = X_train.shape[1]
        batch_size =128
        print("Counter test")
        from collections import Counter
        print(Counter(y_train))

        y_train = np.array(y_train)
        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train))
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

                loss, acc = model.train_on_batch(x, y_temp, class_weight=class_weights)
                #print (loss, acc)
                
        temp = model.predict_on_batch(X_test)
        y_pred=[]
        for i in temp:
            if i[0] > 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
        print (classification_report(y_test, y_pred))
        print (precision_recall_fscore_support(y_test, y_pred))
                
        X_train, y_train = gen_data(tweets_train,word2vec_model,flag)
        X_test, y_test = gen_data(tweets_test,word2vec_model,flag)
        
        wordEmb = model.layers[0].get_weights()[0]
        precision, recall, f1_score, acc, p_weighted, p_macro, r_weighted, r1_macro, f1_weighted, f11_macro = gradient_boosting_classifier([], wordEmb,[],[], X_train, y_train, X_test, y_test,flag)
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
    flag = 'binary'#parser.parse_args().type
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_MODEL_FILE)
    run_exp_6(flag, None)
