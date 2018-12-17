#LSTM Badjatiya model, modified in lines 10 y 11
#Experiment 6
def lstm_model(sequence_length, embedding_dim):
    model_variation = 'LSTM'
    print('Model variation is %s' % model_variation)
    model = Sequential()
    model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length, trainable=LEARN_EMBEDDINGS))
    model.add(Dropout(0.25))
    model.add(LSTM(embedding_dim))
    model.add(Dropout(0.5))
    model.add(Dense(1))#model.add(Dense(3))
    model.add(Activation('sigmoid'))#model.add(Activation('softmax'))
    model.compile(loss=LOSS_FUN, optimizer=OPTIMIZER, metrics=['accuracy'])
    print (model.summary())
    return model