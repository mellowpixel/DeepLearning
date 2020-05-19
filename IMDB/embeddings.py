#! bin/python3
import os
import tensorflow as tf
import tensorflow_hub as hub

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
    dataset_orig = dataset.copy()
    
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
    stats_bank = {"embeddings":[], "word2vec":[], "nnlm":[]}
    max_words   = n_top_used_words
    vector_size = 32
    input_size  = X_train.shape[1]
    batch_size  = 100
    epochs      = 5

    # **************************************************************************************** #
    #  DATASET PREPROCESSING TO CREATE 1D TENSOR OF STRINGS REQUIRED FOR PRETRAINED EMBEDDINGS #
    # **************************************************************************************** #


    strt1d_dataset = prep.string_to_int(df=dataset_orig, params={"sentiment": {'positive': 1, 'negative': 0}})
    strt1d_train, strt1d_test, strt1d_eval = prep.split_dataset(training_r=0.5, test_r=0.3, validation_r=0.2, dataset=strt1d_dataset)
    
    X_strt1d_train = strt1d_train["review"]
    Y_strt1d_train = np.array(strt1d_train['sentiment'].tolist())
    
    X_strt1d_eval = strt1d_eval["review"]
    Y_strt1d_eval = np.array(strt1d_eval['sentiment'].tolist())

    X_strt1d_test = strt1d_test["review"]
    Y_strt1d_test = np.array(strt1d_test['sentiment'].tolist())
    

    # ******************************************************* #
    #  NEURAL NETWORK LANGUAGE TOKEN EMBEDDINGS MODEL (NNLM)  #
    # ******************************************************* #
    
    nnlm_stats  = Stats()
    # If the module doesn't download automatically, then please download it manualy via the curl request and provide 
    # the hub.KerasLayer with the path to the module folder
    # curl -L "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2?tf-hub-format=compressed" | tar -zxvC tf_hub_modules/nnlm

    nnlm_model= tf.keras.Sequential([
        # hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2", input_shape=[], dtype=tf.string, trainable=False),
        hub.KerasLayer("tf_hub_modules/nnlm", input_shape=[], dtype=tf.string, trainable=False),
        tf.keras.layers.Dense(128, activation=tf.nn.relu, use_bias=True),
        tf.keras.layers.Dense(64, activation=tf.nn.relu, use_bias=True),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    nnlm_model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    print(nnlm_model.summary())
    nnlm_model.fit(X_strt1d_train, Y_strt1d_train, batch_size=batch_size, epochs=epochs, validation_data=(X_strt1d_eval, Y_strt1d_eval), callbacks=[nnlm_stats])
    stats_bank['nnlm'] = nnlm_stats.training

    # ************************************************** #
    #          Word2Vec Wiki words 550 MODEL             #
    # ************************************************** #
    
    word2vec_stats  = Stats()

    word2vec_model= tf.keras.Sequential([
        hub.KerasLayer("tf_hub_modules/word2vec", input_shape=[], dtype=tf.string, trainable=False),
        tf.keras.layers.Dense(128, activation=tf.nn.relu, use_bias=True),
        tf.keras.layers.Dense(64, activation=tf.nn.relu, use_bias=True),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    word2vec_model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    print(word2vec_model.summary())
    word2vec_model.fit(X_strt1d_train, Y_strt1d_train, batch_size=batch_size, epochs=epochs, validation_data=(X_strt1d_eval, Y_strt1d_eval), callbacks=[word2vec_stats])
    stats_bank['word2vec'] = word2vec_stats.training

    # ************************************************** #
    #            Distributed Embeddings MODEL            #
    # ************************************************** #
    
    embedding_stats  = Stats()

    embeddings_model = Sequential([
        Embedding(max_words + 1, output_dim=vector_size, input_length=input_size, batch_input_shape=[batch_size, None]),
        Flatten(),
        Dense(128, activation=tf.nn.relu, use_bias=True),
        Dense(64, activation=tf.nn.relu, use_bias=True),
        Dense(1, activation=tf.nn.sigmoid)
    ])

    embeddings_model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    for i in range(epochs):
        embeddings_model.fit(X_train, Y, epochs=1, batch_size=batch_size, validation_data=(X_eval, Yv), callbacks=[embedding_stats])
        stats_bank['embeddings'].append(embedding_stats.training[-1])
        embeddings_model.reset_states()    

    print(embeddings_model.summary())

    # ************************************************** #
    #                EVALUATION OF THE MODELS            #
    # ************************************************** #

    print("\n* Evaluating accuracy of the Distributed Embedding model")
    accuracy = embeddings_model.evaluate(X_test, Yt, verbose=1, batch_size=batch_size, callbacks=[embedding_stats])
    print("Accuracy: {}%".format(round(accuracy[1]*100, 2)))

    print("\n* Evaluating accuracy of the Word2Vec model")
    w2v_accuracy = word2vec_model.evaluate(X_strt1d_test, Y_strt1d_test, batch_size=batch_size, verbose=1, callbacks=[word2vec_stats])
    print("Accuracy: {}%".format(round(w2v_accuracy[1]*100, 2)))

    print("\n* Evaluating accuracy of the NNLM model")
    nnlm_accuracy = nnlm_model.evaluate(X_strt1d_test, Y_strt1d_test, batch_size=batch_size, verbose=1, callbacks=[nnlm_stats])
    print("Accuracy: {}%".format(round(nnlm_accuracy[1]*100, 2)))

    stats_o = {
        "Word2Vec": {
            "training":stats_bank['word2vec'], 
            "test":word2vec_stats.test
        },
        "NNLM": {
            "training":stats_bank['nnlm'], 
            "test":nnlm_stats.test
        }, 
        "Distributed Embedding": {
            "training":stats_bank['embeddings'], 
            "test":embedding_stats.test
            }
        }

    Stats().show_training_stats(stats_o)
    Stats().show_test_stats(stats_o)
    Stats().training_performance(stats_o)
    plt.show()
