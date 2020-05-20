#! bin/python3
import os
import tensorflow as tf
# from tensorflow.keras.models import load_model

from keras.models import Sequential, load_model
from keras.layers import Input, SimpleRNN, LSTM, Embedding, Dense, Flatten
from keras import optimizers
from keras import metrics
from keras import losses
import keras.backend as K

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
    dataset = pd.read_csv(dir + '/imdb_reviews_1983.csv')

    stats = Stats()
    prep = Preprocessing(dataset)
    
    # Make a dictionary by tokenizing all words in the dataset
    prep.make_dictionary()
    
    # Encode all words with integer IDs
    # Encode only the most used words in the dataset, any other words encode as 0
    dataset = prep.encode_dataset_column(df=dataset, field="review")

    # Encode target variables to binary representation
    dataset = prep.string_to_int(df=dataset, params={"sentiment": {'positive': 1, 'negative': 0, 'Positive': 1, 'Negative': 0}})

    # Pad all reviews, remove reviews that have no words, trim reviews that exceed the review_len value
    review_len = 500
    dataset = prep.pad_text(df=dataset, column="review_encoded", min_words=1, max_words=review_len)

    # Split the dataset into training and validation subsets leave test set empty
    train_s, _, valid_s = prep.split_dataset(training_r=0.7, test_r=0.0, validation_r=0.3, dataset=dataset)

    # Convert dataframe column to the numpy array
    X_train = np.array(train_s['review_encoded'].tolist())
    Y = np.array(train_s['sentiment'].tolist())

    X_train = np.concatenate((X_train, np.zeros((230, 500), dtype=int)))
    Y = np.concatenate((Y, np.zeros(230, dtype=int)))

    X_eval = np.array(valid_s['review_encoded'].tolist())
    Yv = np.array(valid_s['sentiment'].tolist())

    X_eval = np.concatenate((X_eval, np.zeros((242, 500), dtype=int)))
    Yv = np.concatenate((Yv, np.zeros(242, dtype=int)))

    # ************************************************** #
    #                THE SIMPLE RNN MODEL                #
    # ************************************************** #

    state_size  = 128
    max_words   = 10000
    vector_size = 32
    input_size  = X_train.shape[1]
    batch_size  = 50
    epochs      = 4

    stats  = Stats()

    model = Sequential([
        Embedding(max_words + 1, output_dim=vector_size, input_length=input_size, batch_input_shape=[batch_size, None]),
        SimpleRNN(state_size, return_sequences=True, stateful=True, activation=tf.nn.relu),
        Flatten(),
        Dense(64, activation=tf.nn.sigmoid),
        Dense(1,   activation=tf.nn.sigmoid)
    ])

    model.load_weights("saved_model/simple_RNN_weights.h5", by_name=True)

    model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    for i in range(epochs):
        model.fit(X_train, Y, epochs=1, batch_size=batch_size, validation_data=(X_eval, Yv), callbacks=[stats])
        model.reset_states()

    print(model.summary())
    

    # ************************************************** #
    #                   Saving the model                 #
    # ************************************************** #

    # model.save("saved_model/best-model-save.h5", overwrite=True, include_optimizer=True)
    # model.save_weights("saved_model/simple_RNN_weights.h5")

    # ************************************************** #
    #                   Show Statistics                  #
    # ************************************************** #

    stats_o = {
        "Simple RNN": {
            "training": stats.training, 
            "test": stats.test
        }
    }

    Stats().show_training_stats(stats_o)
    Stats().show_test_stats(stats_o)
    plt.show()