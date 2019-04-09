import argparse
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pdb
from auxiliares import *
vocab, reverse_vocab = {}, {}

def run_exp_3 ():
    #Experimento Cross Domain Waseem and Hovy dataset as trainning set and SemEval dataset as testing set
    tweets_train,_ = select_tweets('waseem',None)
    
    tweets_test ,_ = select_tweets('sem_eval',None)
    
    vocab = gen_vocab(np.concatenate((tweets_train,tweets_test), axis = 0))   
    
    MAX_SEQUENCE_LENGTH = max_len(tweets_train)
    
    train_LSTM_Cross_Domain(tweets_train, tweets_test, vocab, MAX_SEQUENCE_LENGTH) 
    
    
def train_LSTM_Cross_Domain(tweets_train, tweets_test, vocab, MAX_SEQUENCE_LENGTH):
        a, p, r, f1 = 0., 0., 0., 0.
        a1, p1, r1, f11 = 0., 0., 0., 0.
        pn,rn,fn = 0.,0.,0.
        sentence_len = MAX_SEQUENCE_LENGTH
        batch_size =128
        
        X_train, y_train = gen_sequence(tweets_train,vocab,'categorical')
        X_test, y_test = gen_sequence(tweets_test,vocab,'binary')
        
        X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
        X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

        
        y_train = np.array(y_train)
        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train))
        
        
        model = lstm_model(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

        if INITIALIZE_WEIGHTS_WITH == "glove":
            weights = get_embedding_weights(vocab)
            model.layers[0].set_weights([weights])
        elif INITIALIZE_WEIGHTS_WITH == "random":
            shuffle_weights(model)
        else:
            print ("ERROR!")
            return
        for epoch in range(EPOCHS):
            for X_batch in batch_gen(X_temp, BATCH_SIZE):
                x = X_batch[:, :sentence_len]
                y_temp = X_batch[:, sentence_len]

                try:
                    y_temp = np_utils.to_categorical(y_temp, num_classes=3)
                except Exception as e:
                    print (e)
                #print (x.shape, y_temp.shape)
                loss, acc = model.train_on_batch(x, y_temp, class_weight=None)
                #print (loss, acc)
                
        temp = model.predict_on_batch(X_test)
        y_pred_aux = np.argmax(temp, axis=1)
        y_pred=[]
        for i in y_pred_aux:
            if i == 2:
                y_pred.append(1)
            else:
                y_pred.append(i)
                
#         print (classification_report(y_test, y_pred))
#         print (precision_recall_fscore_support(y_test, y_pred))

        wordEmb = model.layers[0].get_weights()[0]

        word2vec_model = create_model(wordEmb,vocab)
        
        tweets_train = select_tweets_whose_embedding_exists(tweets_train, word2vec_model)
        tweets_test = select_tweets_whose_embedding_exists(tweets_test, word2vec_model)
        
        X_train, y_train = gen_data(tweets_train,word2vec_model,'categorical')
        X_test, y_test = gen_data(tweets_test,word2vec_model,'binary')
        
        precision, recall, f1_score, acc, p_weighted, p_macro, r_weighted, r1_macro, f1_weighted, f11_macro = gradient_boosting_classifier(X_train, y_train, X_test, y_test, 'cross')
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
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_MODEL_FILE)

    run_exp_3()
