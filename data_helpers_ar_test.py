#This open source code is from a project in github https://github.com/dennybritz/cnn-text-classification-tf
import os
import string
import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # Clean up the data and replace the word-free symbol
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(AA_data_file, BB_data_file, CC_data_file, DD_data_file):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files  #Split sample
    A_examples = list(open(AA_data_file, "r").readlines())
    A_examples = [s.strip() for s in A_examples]
    B_examples = list(open(BB_data_file, "r").readlines())
    B_examples = [s.strip() for s in B_examples] 
    C_examples = list(open(CC_data_file, "r").readlines())
    C_examples = [s.strip() for s in C_examples]
    D_examples = list(open(DD_data_file, "r").readlines())
    D_examples = [s.strip() for s in D_examples] 

    # Split by words  
    x_text = A_examples + B_examples + C_examples + D_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels 
    A_labels = [[1, 0 ,0 ,0] for _ in A_examples]
    B_labels = [[0, 1 ,0 ,0] for _ in B_examples]
    C_labels = [[0, 0 ,1 ,0] for _ in C_examples]
    D_labels = [[0, 0 ,0 ,1] for _ in D_examples]
    y = np.concatenate([A_labels,B_labels,C_labels,D_labels],0)
    return [x_text, y]


# Creating an batch iteration module
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    # everytime print shuffled_data[start_index:end_index]
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))# shuffle
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size # begin from the current batch
            end_index = min((batch_num + 1) * batch_size, data_size)# Determine whether the next batch will exceed the last data.
            yield shuffled_data[start_index:end_index]
