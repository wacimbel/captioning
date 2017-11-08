# Function for handling the dataset
import os
import string
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
import pickle
import skimage
import skimage.io
import skimage.transform
import random
import json

class Batcher():

    def __init__(self, data_path, config, vocab):
        """
        Creates an object based on the path of the images and annotations. The batcher will provide batches of data
        during training, containing both images and annotations. It will also count down the iterations and epochs that has been made
        """

        self.train_path = os.path.join(data_path, 'train/')

        self.batch_size = config['batch_size']
        self.im_width = config['im_width']
        self.im_height = config['im_height']
        self.max_len = config['nb_LSTM_cells']
        self.vocab = vocab
        self.current_idx = 0
        self.epoch_completed = 0
        
    def load_annotations(self):
        annot = json.load(open(self.train_path+'annotations/captions_train2014.json', 'r'))
        available_im = os.listdir(self.train_path+'images/')
        self.ids = [elem['id'] for elem in annot['images'] if elem['file_name'] in available_im]
        
        self.nb_ids = len(self.ids)
        #random.shuffle(self.ids)
        
        captions_list = [elem for elem in annot['annotations'] if elem['image_id'] in self.ids]
        self.captions = pd.DataFrame(captions_list).groupby('image_id')['caption'].apply(list)

    
    def load_image(self, path):
        # load image
        img = skimage.io.imread(path)
        img = img / 255.0
        assert (0 <= img).all() and (img <= 1.0).all()
        # print "Original Image Shape: ", img.shape
        # we crop image from center
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
        # resize to 224, 224
        resized_img = skimage.transform.resize(crop_img, (self.im_width, self.im_height))
        return resized_img

    def next_batch(self):
        """
        :return: a batch containing images and their encoded annotations, picked in the paths folder. Deals with
        the number of completed epochs.
        Images is of shape (batch_size, im_width, im_height)
        Annotations is of shape (batch_size, max_length)
        """
        next_idx = self.current_idx+4
        batch_ids = self.ids[self.current_idx:next_idx]
        
        if next_idx > self.nb_ids:
            self.epoch_completed +=1
            random.shuffle(self.ids)
            remaining = next_idx - self.nb_ids
            batch_ids += self.ids[:remaining]
            next_idx = remaining
            
        imgs = np.zeros((self.batch_size, self.im_width, self.im_height, 3), dtype=np.float)
        labels = np.zeros((self.batch_size, 1), dtype=np.float)
        
        for image_id in batch_ids:
            batch_idx = image_id % self.batch_size
            print(image_id)
            img_name = 'COCO_train2014_000000' + str(image_id) + '.jpg'
            imgs[batch_idx,...] = self.load_image(self.train_path + 'images/' + img_name)
            labels[batch_idx,...] = self.make_labels(self.captions[image_id])
        
        self.current_idx = next_idx
        return imgs, labels
        
    
    def make_labels(self, captions_list):
        return 1

    @staticmethod
    def clean_sentence(sentences):
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
