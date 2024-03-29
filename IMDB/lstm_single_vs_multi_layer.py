#! bin/python3
import os
import tensorflow as tf

from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Embedding, Dense, Flatten
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
    
    stats = Stats()
    prep = Preprocessing(dataset)
    
    # Make a dictionary by tokenizing all words in the dataset
    prep.make_dictionary()
    
    # Encode all words with integer IDs
    # Encode only the most used words in the dataset, any other words encode as 0
    n_top_used_words = 10000
    dataset = prep.encode_dataset_column(df=dataset, field="review", use_top_words=n_top_used_words)

    # Encode target variables to binary representation
    dataset = prep.string_to_int(df=dataset, params={"sentiment": {'positive': 1, 'negative': 0}})

    # Pad all reviews, remove reviews that have no words, trim reviews that exceed the review_len value
    review_len = 500
    dataset = prep.pad_text(df=dataset, column="review_encoded", min_words=1, max_words=review_len)

    # Split the dataset into training, test and validation subsets
    train_s, test_s, valid_s = prep.split_dataset(training_r=0.5, test_r=0.3, validation_r=0.2, dataset=dataset)

    # Convert dataframe column to the numpy array
    X_train = np.array(train_s['review_encoded'].tolist())
    Y = np.array(train_s['sentiment'].tolist())

    X_eval = np.array(valid_s['review_encoded'].tolist())
    Yv = np.array(valid_s['sentiment'].tolist())

    X_test = np.array(test_s['review_encoded'].tolist())
    Yt = np.array(test_s['sentiment'].tolist())

    # ************************************************** #
    #              MODELS COMMON SETTINGS                #
    # ************************************************** #
    stats_bank = {"lstm_single":[], "lstm_multi":[]}

    state_size  = 128
    max_words   = n_top_used_words
    vector_size = 32
    input_size  = X_train.shape[1]
    batch_size  = 250
    epochs      = 3

    # ************************************************** #
    #            THE Single-layer LSTM MODEL             #
    # ************************************************** #
    
    lstm_single_stats  = Stats()

    lstm_single_model=Sequential([
        Embedding(max_words + 1, output_dim=vector_size, input_length=input_size, batch_input_shape=[batch_size, None]),
        LSTM(state_size, return_sequences=True, stateful=True, activation=tf.nn.tanh, dropout=0.2),
        Flatten(),
        # Dense(64, activation=tf.nn.sigmoid),
        Dense(1, activation=tf.nn.sigmoid)
    ])

    lstm_single_model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    for i in range(epochs):
        lstm_single_model.fit(X_train, Y, epochs=1, batch_size=batch_size, validation_data=(X_eval, Yv), callbacks=[lstm_single_stats])
        stats_bank['lstm_single'].append(lstm_single_stats.training[-1])
        lstm_single_model.reset_states()    

    print(lstm_single_model.summary())

    # ************************************************** #
    #                Multi-layer LSTM MODEL              #
    # ************************************************** #

    lstm_multi_stats  = Stats()

    lstm_multi_model=Sequential([
        Embedding(max_words + 1, output_dim=vector_size, input_length=input_size, batch_input_shape=[batch_size, None]),
        LSTM(state_size, return_sequences=True, stateful=True, activation=tf.nn.tanh, dropout=0.2),
        LSTM(64, return_sequences=True, stateful=True, activation=tf.nn.tanh, dropout=0.2),
        Flatten(),
        # Dense(64, activation=tf.nn.sigmoid),
        Dense(1,   activation=tf.nn.sigmoid)
    ])

    lstm_multi_model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    # print(lstm_multi_model.summary())
    
    for i in range(epochs):
        lstm_multi_model.fit(X_train, Y, epochs=1, batch_size=batch_size, validation_data=(X_eval, Yv), callbacks=[lstm_multi_stats])
        stats_bank['lstm_multi'].append(lstm_multi_stats.training[-1])
        lstm_multi_model.reset_states()

    print(lstm_multi_model.summary())

    # ************************************************** #
    #                EVALUATION OF THE MODELS            #
    # ************************************************** #

    print("\n* Evaluating accuracy of the Single Layer LSTM model")
    accuracy = lstm_single_model.evaluate(X_test, Yt, verbose=1, batch_size=batch_size, callbacks=[lstm_single_stats])
    print("Accuracy: {}%".format(round(accuracy[1]*100, 2)))

    print("\n* Evaluating accuracy of the Multi Layer LSTM model")
    lstm_m_accuracy = lstm_multi_model.evaluate(X_test, Yt, batch_size=batch_size, verbose=1, callbacks=[lstm_multi_stats])
    print("Accuracy: {}%".format(round(lstm_m_accuracy[1]*100, 2)))

    stats_o = {
        "Multi-layer LSTM": {
            "training":stats_bank['lstm_multi'], 
            "test":lstm_multi_stats.test
        }, 
        "Single-layer LSTM": {
            "training":stats_bank['lstm_single'], 
            "test":lstm_single_stats.test
            }
        }

    Stats().show_training_stats(stats_o)
    Stats().show_test_stats(stats_o)
    Stats().training_performance(stats_o)
    plt.show()
