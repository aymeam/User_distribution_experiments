import tflearn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
import tensorflow as tf
import os
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Convolution1D, MaxPooling1D, GlobalMaxPooling1D

#os.environ['KERAS_BACKEND']='theano'
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model,Sequential

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers, optimizers

def lstm_model_bin(sequence_length, embedding_dim,vocab):
    model_variation = 'LSTM'
    print('Model variation is %s' % model_variation)
    model = Sequential()
    print('variables')
    print(embedding_dim)
    print(sequence_length)
    model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length, trainable=True))
    model.add(Dropout(0.25))#, input_shape=(sequence_length, embedding_dim)))
    model.add(LSTM(embedding_dim))#, input_shape=(sequence_length, embedding_dim)))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print (model.summary())
    return model    
    
def lstm_model(sequence_length, embedding_dim,vocab):
    model_variation = 'LSTM'
    print('Model variation is %s' % model_variation)
    model = Sequential()
    print('variables')
    print(embedding_dim)
    print(sequence_length)
    model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length, trainable=True))
    model.add(Dropout(0.25))#, input_shape=(sequence_length, embedding_dim)))
    model.add(LSTM(embedding_dim))#, input_shape=(sequence_length, embedding_dim)))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print (model.summary())
    return model