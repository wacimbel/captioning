import json
from collections import Counter
import string
import pickle

# This is the script used to create the vocab file in this repo. In case the training data changes,
# or our method to generate the vocab, we can reuse this script to generate a new vocabulary file.

# Load raw captions
captions_path = 'coco/train/annotations/captions_train2014.json'
captions = json.load(open(captions_path, 'r'))
raw_captions = [i['caption'] for i in captions['annotations']]

# Clean sentences
counter = Counter()
for caption in raw_captions:
    for sign in string.punctuation:
        caption = caption.replace(sign, ' ')
    counter.update(caption.lower().split())

# Filtering on frequent-enough words
frequency_boundary = 5
vocab = [i[0] for i in counter.most_common(100000000) if i[1] >= frequency_boundary]

# Saving the vocab file in the current folder
filename = 'vocab'
pickle.dump(vocab, open(filename, 'wb'))