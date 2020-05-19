import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

class Preprocessing():
    
    dictionary = {}

    def __init__(self, dataset):
        """ Constructor """
        self.dataset = dataset
        print("\n* Total records in the dataset: ", len(dataset))

    

    def string_to_int(self, df, params):
        """ Replace given string values in the dataset with an integer """
        
        df = df.replace(params, inplace = False)

        return df



    def make_dictionary(self):
        """ Tokenize all words in the dataset """
        words_count = {}

        for review in self.dataset['review']:
            # Remove all punctuation and split the review to separate words
            words = self.text_to_words(review) 
            # Collect all unique words and count the number of times a word appears in the dataset
            for word in words:
                if word in words_count:
                    words_count[word] += 1
                else:
                    words_count[word] = 1
        
        self.words_counted = words_count
        # Sort words by their count
        self.dictionary = [ word for word, count in sorted(words_count.items(), key=lambda item: item[1], reverse=True) ]
        # Tokenize all words in the dictionary
        self.dictionary = { word:i for i, word in enumerate(self.dictionary, 1)}

        print("\n-----------------------------------------------------")
        print("* Total words in the dictionary: ", len(self.dictionary))
        print("* Dictionary head 5: ", dict(list(self.dictionary.items())[0:10]))
        print("* Dictionary tail 5: ", dict(list(self.dictionary.items())[-5:-1]))

        return self.dictionary



    def split_dataset(self, training_r=0.5, test_r=0.3, validation_r=0.2, dataset=None):
        """ Splits the dataset into the given ratios"""
        
        if dataset is None:
            dataset = self.dataset

        # Calculate sets sizes
        total_rows  = len(dataset)
        train_size = int(total_rows * training_r)
        test_size  = int(total_rows * test_r)
        valid_size = int(total_rows * validation_r)

        # Create random indices for each of the sets
        rand_indeces   = np.random.randint(low=0,high=total_rows-1, size = total_rows)
        train_indeces  = rand_indeces[0 : train_size]
        test_indeces   = rand_indeces[train_size : train_size + test_size]
        valid_indeces  = rand_indeces[train_size + test_size : train_size + test_size +valid_size]

        # Locate data at the indeces and copy it into the subsets
        training_set   = dataset.iloc[train_indeces, :].reset_index()
        test_set       = dataset.iloc[test_indeces,  :].reset_index()
        validation_set = dataset.iloc[valid_indeces, :].reset_index()

        print("\n-----------------------------------------------------")
        print("* Training set size: {}% = {} records".format(training_r * 100, train_size))
        print("* Test set size: {}% = {} records".format(test_r * 100, test_size))
        print("* Validation set size: {}% = {} records".format(validation_r * 100, valid_size))

        return training_set, test_set, validation_set


    
    def encode_dataset_column(self, df=None, field=None, use_top_words=None):
        encoded_data = []
        
        if df is None:
            df = self.dataset

        for text in df[field]:
            encoded_text = self.encode_text(text, use_top_words=use_top_words)
            encoded_data.append(encoded_text)

        df[field+"_encoded"] = encoded_data

        print("\n-----------------------------------------------------")
        print("* Reviews encoded.")

        return df



    def encode_text(self, text, use_top_words=None):
        encoded_text = []
        words = self.text_to_words(text)

        for word in words:
            word_enc = self.dictionary[word]
            
            if use_top_words is not None and word_enc > use_top_words:
                word_enc = 0
            
            encoded_text.append(word_enc)

        return encoded_text



    def text_to_words(self, text):
        text = re.sub(r"<.*?>|[^a-zA-Z0-9\s-]", "", text.lower()) # Remove all punctuation
        words = text.split() # Split the string into separate words

        return words



    def pad_text(self, df, column, max_words, min_words):

        for i, _ in df.iterrows():
            l = len(df[column][i])
            
            # Truncate the text that is longer then max_words
            if l > max_words:
                df[column][i] = df[column][i][0:max_words]

            # Remove reviews whos word count isless then the alowed minimum
            if l < min_words:
                df.drop(index=i, inplace=True)

            # Pad/prepend the text with 0 if it is shorter then the max_words
            if l < max_words:
                df[column][i] = df[column][i] + [0 for _ in range(0, max_words - l)]

        return df
