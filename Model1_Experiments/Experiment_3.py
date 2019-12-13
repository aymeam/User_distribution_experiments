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
    
   # vocab = gen_vocab(np.concatenate((tweets_train,tweets_test), axis = 0))   
    vocab = gen_vocab(tweets_train)   

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
        
        
        model = lstm_model(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,vocab)
#        shuffle_weights(model)
        model.layers[0].set_weights([get_embedding_weights2(vocab)])

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
        
      
        X_train, y_train = gen_data(tweets_train,word2vec_model,'categorical')
        X_test, y_test = gen_data(tweets_test,word2vec_model,'binary')
        
        model = gradient_boosting_classifier(X_train, y_train)
        precision, recall, f1_score,precisionw, recallw, f1_scorew,precisionm, recallm, f1_scorem =evaluate_model(model, X_test, y_test, 'cross')
        
        p += precisionw
        p1 += precisionm
        r += recallw
        r1 += recallm
        f1 += f1_scorew
        f11 += f1_scorem
        pn += precision
        rn += recall
        fn += f1_score
        print_scores(p, p1, r,r1, f1, f11,pn, rn, fn,1)

    
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
    run_exp_3()