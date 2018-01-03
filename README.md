# Image captioning

Architecture: CNN to extract features in the image and RNN to generate captions.

Folder coco: contains part of the dataset to make unit test. The full dataset is retrieved with direct link when training on AWS.

create_vocab.py: create the vocabulary according to all the training captions

data.py: the whole preprocessing of the dataset (images and captions)

model.py: the model in Tensorflow and the training setup

run_training.py: main file to run the training
