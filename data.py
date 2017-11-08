# Function for handling the dataset
import os
import string
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
import pickle

class Batcher():

    def __init__(self, data_path, config, vocab):
        """
        Creates an object based on the path of the images and annotations. The batcher will provide batches of data
        during training, containing both images and annotations. It will also count down the iterations and epochs that has been made
        """

        self.images_path = os.path.join(data_path, 'images')
        self.annotations_path = os.path.join(data_path, 'annotations')

        self.batch_size = config['batch_size']
        self.im_width = config['im_width']
        self.im_height = config['im_height']
        self.max_len = config['nb_LSTM_cells']
        self.vocab = vocab
        self.epoch_completed = 0

    def resize(self, image):
        """
        :param image: image to be resized
        :return: resized image at the dimension self.im_width, self.im_height
        """

        return resized_image

    def next_batch(self):
        """
        :return: a batch containing images and their encoded annotations, picked in the paths folder. Deals with
        the number of completed epochs
        """

        return images, annotations

    @staticmethod
    def clean_sentence(sentence):
        # This method has been put as static as we might need it for other purposes outside the particular instance
        # of the class
        """
        :return: Cleans the sentence passed in argument
        """
        for sign in string.punctuation:
            sentence = sentence.replace(sign, ' ')

        return sentence.lower()

    def encode_sentence(self, sentence, vocab):
        """

        :param sentence: Raw sentence to encode
        :param vocab: global vocab file.
        :return: Encoded sentence with IDs of the words, padded to a fixed size length.
        """
        ids = [vocab.get_word_index(word) for word in self.clean_sentence(sentence).split()]

        return pad_sequences(ids, maxlen=self.max_len)


class Vocab():
    """
    This class aims at loading the french vocabulary, and implements a couple of convenient functions to deal with
    the vocabulary for embeddings. We load a vocab file which structure is: {id_word: (word, word_count)}
    """

    def __init__(self, savepath):
        self.savepath = savepath
        self.vocab = pickle.load(open(self.savepath, 'rb'))
        self.size = len(self.vocab)
        self.word_to_index = {}
        self.index_to_counts = {}
        self.index_to_word = {}
        for i in self.vocab.keys():
            self.index_to_counts[i] = self.vocab[i][1]
            self.word_to_index[self.vocab[i][0]] = i
            self.index_to_word[i] = self.vocab[i][0]

    def get_word_count(self, word):
        return self.vocab[self.word_to_index[word]][1]

    def get_word_index(self, word):
        return self.word_to_index[word]

    def get_index_count(self, index):
        return self.index_to_counts[index]

    def get_index_word(self, index):
        return self.index_to_word[index]
