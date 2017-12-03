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
import tensornets as nets


class Batcher():

    def __init__(self, data_path, config, vocab):
        """
        Creates an object based on the path of the images and annotations. The batcher will provide batches of data
        during training, containing both images and annotations. It will also count down the iterations and epochs that has been made
        """

        self.train_path = os.path.join(data_path, 'train/')
        self.val_path = os.path.join(data_path, 'val/')
        self.batch_size = config['batch_size']
        self.im_width = config['im_width']
        self.im_height = config['im_height']
        self.max_len = config['nb_LSTM_cells']
        self.vocab = vocab
        self.current_idx = 0
        self.epoch_completed = 0
        self.load_train_annotations()
        self.load_val_annotations()

    def load_train_annotations(self):
        annot = json.load(open(self.train_path+'annotations/captions_train2014.json', 'r'))
        available_im = os.listdir(self.train_path+'images/')
        self.train_ids = [elem['id'] for elem in annot['images'] if elem['file_name'] in available_im]
        
        self.nb_ids = len(self.train_ids)
        
        captions_list = [elem for elem in annot['annotations'] if elem['image_id'] in self.train_ids]
        #self.captions = pd.DataFrame(captions_list).groupby('image_id')['caption'].apply(list)
        self.train_captions = pd.DataFrame(captions_list).set_index('image_id')

        
    def load_val_annotations(self):
        annot = json.load(open(self.val_path+'annotations/captions_val2014.json', 'r'))
        available_im = os.listdir(self.val_path+'images/')
        self.val_ids = [elem['id'] for elem in annot['images'] if elem['file_name'] in available_im]


        
        captions_list = [elem for elem in annot['annotations'] if elem['image_id'] in self.val_ids]
        #self.captions = pd.DataFrame(captions_list).groupby('image_id')['caption'].apply(list)
        self.val_captions = pd.DataFrame(captions_list).set_index('image_id')
    
    def load_image(self, path):
        # load image
        img = skimage.io.imread(path)
        img = img / 255.0
        assert (0 <= img).all() and (img <= 1.0).all()
        # print "Original Image Shape: ", img.shape
        # we crop image from center
        long_edge = max(img.shape[:2])
        padded_img = np.zeros((long_edge, long_edge, 3))
        yy = int((long_edge - img.shape[0]) / 2)
        xx = int((long_edge - img.shape[1]) / 2)
        padded_img[yy: yy + img.shape[0], xx: xx + img.shape[1]] = img
        # resize to 224, 224
        resized_img = skimage.transform.resize(padded_img, (self.im_width, self.im_height))        
        return resized_img

    def next_train_batch(self, model=None):
        """
        :return: a batch containing images and their encoded annotations, picked in the paths folder. Deals with
        the number of completed epochs.
        Images is of shape (batch_size, im_width, im_height)
        Annotations is of shape (batch_size, max_length)
        """
        next_idx = self.current_idx+self.batch_size
        batch_ids = self.train_ids[self.current_idx:next_idx]
        
        if next_idx > self.nb_ids:
            self.epoch_completed +=1
            random.shuffle(self.train_ids)
            remaining = next_idx - self.nb_ids
            batch_ids += self.train_ids[:remaining]
            next_idx = remaining
            
        imgs = np.zeros((self.batch_size, self.im_width, self.im_height, 3), dtype=np.float)
        labels = np.zeros((self.batch_size, self.max_len), dtype=np.int)
        
        for i, image_id in enumerate(batch_ids):
            batch_idx = i % self.batch_size
            img_name = 'COCO_train2014_000000' + str(image_id) + '.jpg'
            imgs[batch_idx, ...] = self.load_image(self.train_path + 'images/' + img_name)
            if model is not None:
                imgs[batch_idx, ...] = nets.preprocess(model, imgs[batch_idx, ...])
            #sentence = self.train_captions.loc[image_id].sample(1)['caption'].values[0]
            #TEMP we select necessarily the same caption
            sentence = self.train_captions.loc[image_id].iloc[0]['caption']
            labels[batch_idx, ...] = self.encode_sentence(sentence, self.vocab)
        
        self.current_idx = next_idx

        return imgs, labels
        
        
    def next_val_batch(self, model):
        
        random.shuffle(self.val_ids)
        batch_ids = self.val_ids[:self.batch_size]
        
        imgs = np.zeros((self.batch_size, self.im_width, self.im_height, 3), dtype=np.float)
        labels = np.zeros((self.batch_size, self.max_len), dtype=np.int)
        
        for i, image_id in enumerate(batch_ids):
            batch_idx = i % self.batch_size
            img_name = 'COCO_val2014_000000' + str(image_id) + '.jpg'
            imgs[batch_idx, ...] = self.load_image(self.val_path + 'images/' + img_name)
            if model is not None:
                imgs[batch_idx, ...] = nets.preprocess(model, imgs[batch_idx, ...])    
            sentences = self.val_captions.loc[image_id].iloc[0]['caption']
            labels[batch_idx, ...] = self.encode_sentence(sentences, self.vocab)

        return imgs, labels
        

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
        ids = vocab.add_start_end(ids)

        return pad_sequences([ids], maxlen=self.max_len, padding='post', truncating='post')


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
        return self.word_to_index.get(word, 1)

    def get_index_count(self, index):
        return self.index_to_counts[index]
        # return self.index_to_counts.get(index, 0)

    def get_index_word(self, index):
        return self.index_to_word[index]
        # return self.index_to_word.get(index, '<UNK>')

    def add_start_end(self, ids):
        return [2] + ids + [3]

