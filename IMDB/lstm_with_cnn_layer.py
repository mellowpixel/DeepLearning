#! bin/python3
import os
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Flatten, Conv1D, MaxPool1D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import optimizers
from keras import metrics
from keras import losses

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from classes.preprocessing import Preprocessing
from classes.stats import Stats



if __name__ == '__main__':
    
    # ************************************************** #
    #           PREPROCESSING OF THE DATASET             #
    # ************************************************** #

    dir = os.path.dirname(os.path.realpath(__file__))
    dataset = pd.read_csv(dir + '/dataset/IMDBDataset.csv')
    
    
    prep = Preprocessing(dataset)
    
    # Make a dictionary by tokenizing all words in the dataset
    prep.make_dictionary()
    
    # Encode all words with integer IDs
    # Encode only the most used words in the dataset, any other words encode as 0
    n_top_used_words = 10000
    dataset = prep.encode_dataset_column(df=dataset, field="review", use_top_words=n_top_used_words)

    # Encode target variables to binary representation
    dataset = prep.string_to_int(df=dataset, params={"sentiment": {'positive': 1, 'negative': 0}})

    # Show reviews distribution by their word count
    # stats.words_count_distribution(df=dataset, column="review_encoded", show=False, title="Reviews distribution by words count.")

    # Pad all reviews, remove reviews that have no words, trim reviews that exceed the review_len value
    review_len = 200
    dataset = prep.pad_text(df=dataset, column="review_encoded", min_words=1, max_words=review_len)

    # Split the dataset into training, test and validation subsets
    train_s, test_s, valid_s = prep.split_dataset(training_r=0.5, test_r=0.3, validation_r=0.2, dataset=dataset)

    # Show reviews distribution by their word count after transformation
    # stats.words_count_distribution(df=train_s, column="review_encoded", title="Reviews distribution by words count modified.")

    # Convert dataframe column to the numpy array
    X_train = np.array(train_s['review_encoded'].tolist())
    Y = np.array(train_s['sentiment'].tolist())

    X_eval = np.array(valid_s['review_encoded'].tolist())
    Yv = np.array(valid_s['sentiment'].tolist())

    X_test = np.array(test_s['review_encoded'].tolist())
    Yt = np.array(test_s['sentiment'].tolist())
    
 
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # X_eval = np.reshape(X_eval, (X_eval.shape[0], X_eval.shape[1], 1))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # ************************************************** #
    #              MODELS COMMON SETTINGS                #
    # ************************************************** #
    stats_bank = {"lstm":[], "lstm_cnn":[]}
    
    state_size  = 256
    max_words   = n_top_used_words
    vector_size = 32
    input_size  = X_train.shape[1]
    batch_size  = 250
    epochs      = 4


    # ************************************************** #
    #              THE CNN LSTM MODEL                    #
    # ************************************************** #

    lstmcnn_stats = Stats()

    lstmcnn_model=Sequential([
        Embedding(max_words + 1, output_dim=vector_size, input_length=input_size, batch_input_shape=[batch_size, None]),
        Conv1D(filters=128, kernel_size=32, padding='same', activation=tf.nn.tanh),
        MaxPooling1D(),
        LSTM(state_size, return_sequences=True, stateful=True, activation=tf.nn.tanh),
        Flatten(),
        Dense(128, activation=tf.nn.relu, use_bias=True),
        Dense(1, activation=tf.nn.sigmoid)
    ])

    # print_layer_config(lstmcnn_model.layers)
    lstmcnn_model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    # Train the model
    for i in range(epochs):
        lstmcnn_model.fit(X_train, Y, epochs=1, batch_size=batch_size, validation_data=(X_eval, Yv), callbacks=[lstmcnn_stats])
        stats_bank['lstm_cnn'].append(lstmcnn_stats.training[-1])
        lstmcnn_model.reset_states() 

    print(lstmcnn_model.summary())

    # ************************************************** #
    #            THE Single-layer LSTM MODEL             #
    # ************************************************** #
    
    lstm_stats  = Stats()

    lstm_model=Sequential([
        Embedding(max_words + 1, output_dim=vector_size, input_length=input_size, batch_input_shape=[batch_size, None]),
        LSTM(state_size, return_sequences=True, stateful=True, activation=tf.nn.tanh, dropout=0.2),
        Flatten(),
        # Dense(64, activation=tf.nn.sigmoid),
        Dense(1, activation=tf.nn.sigmoid)
    ])

    lstm_model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    for i in range(epochs):
        lstm_model.fit(X_train, Y, epochs=1, batch_size=batch_size, validation_data=(X_eval, Yv), callbacks=[lstm_stats])
        stats_bank['lstm'].append(lstm_stats.training[-1])
        lstm_model.reset_states()    

    print(lstm_model.summary())
    
    # ************************************************** #
    #                EVALUATION OF THE MODELS            #
    # ************************************************** #
    
    print("\n* Evaluating accuracy of the CNN LSTM model")
    accuracy = lstmcnn_model.evaluate(X_test, Yt, verbose=1, batch_size=batch_size, callbacks=[lstmcnn_stats])
    print("Accuracy: {}%".format(round(accuracy[1]*100, 2)))

    # Evaluate the model
    print("\n* Evaluating accuracy of the LSTM model")
    accuracy = lstm_model.evaluate(X_test, Yt, verbose=1, batch_size=batch_size, callbacks=[lstm_stats])
    print("Accuracy: {}%".format(round(accuracy[1]*100, 2)))
    
    stats_o = {
        "CNN LSTM": {
            "training": stats_bank['lstm_cnn'], 
            "test": lstmcnn_stats.test
        },

        "LSTM": {
            "training":stats_bank['lstm'], 
            "test":lstm_stats.test
        }
    }
    
    Stats().show_training_stats(stats_o)
    Stats().show_test_stats(stats_o)
    Stats().training_performance(stats_o)
    plt.show()
