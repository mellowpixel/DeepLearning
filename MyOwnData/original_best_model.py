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
    dataset = pd.read_csv(dir + '/../IMDB/dataset/IMDBDataset.csv')
    
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
    #                THE SIMPLE RNN MODEL                #
    # ************************************************** #

    state_size  = 128
    max_words   = n_top_used_words
    vector_size = 32
    input_size  = X_train.shape[1]
    batch_size  = 250
    epochs      = 3

    simple_stats  = Stats()
    stats_bank = []

    simple_model=Sequential([
        Embedding(max_words + 1, output_dim=vector_size, input_length=input_size, batch_input_shape=[batch_size, None]),
        SimpleRNN(state_size, return_sequences=True, stateful=True, activation=tf.nn.relu),
        Flatten(),
        Dense(64, activation=tf.nn.sigmoid),
        Dense(1,   activation=tf.nn.sigmoid)
    ])

    simple_model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    print(simple_model.summary())
    
    for i in range(epochs):
        simple_model.fit(X_train, Y, epochs=1, batch_size=batch_size, validation_data=(X_eval, Yv), callbacks=[simple_stats])
        stats_bank.append(simple_stats.training[-1])
        simple_model.reset_states()


    # ************************************************** #
    #                EVALUATION OF THE MODELS            #
    # ************************************************** #

    print("\n* Evaluating accuracy of the Simple RNN model")
    simple_accuracy = simple_model.evaluate(X_test, Yt, batch_size=batch_size, verbose=1, callbacks=[simple_stats])
    print("Accuracy: {}%".format(round(simple_accuracy[1]*100, 2)))

    # ************************************************** #
    #                   Saving the model                 #
    # ************************************************** #

    # simple_model.save("saved_model/best-model-save.h5", overwrite=True, include_optimizer=True)
    # simple_model.save_weights("saved_model/simple_RNN_weights.h5")

    # ************************************************** #
    #                   Show Statistics                  #
    # ************************************************** #

    stats_o = {
        "Simple RNN": {
            "training":stats_bank, 
            "test":simple_stats.test
        }
    }

    Stats().show_training_stats(stats_o)
    Stats().show_test_stats(stats_o)
    plt.show()