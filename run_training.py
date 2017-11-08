# Training loop
import json
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from .model import CaptioningNetwork
from .data import Batcher, Vocab


def setup_training(train_dir):
    """

    :param train_dir: Directory where the model and training checkpoints will be saved
    :return: A summary writer object, configured and ready to be used in the training loop
    """
    saver = tf.train.Saver(max_to_keep=3)  # keep 3 checkpoints at a time
    sv = tf.train.Supervisor(logdir=train_dir,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,  # save summaries for tensorboard every 60 secs
                             save_model_secs=60,  # checkpoint every 60 secs
                             global_step=model.global_step)

    summary_writer = sv.summary_writer

    return summary_writer

if __name__ == "__main__":

    config = json.load(open('config.json', 'r'))
    model = CaptioningNetwork(config)
    data_path = 'coco/'
    train_dir = 'Caption_training' + datetime.datetime.strftime(datetime.datetime.today(), '%d%m%Y%H%M%S')
    vocab = Vocab('vocab')
    batcher = Batcher(data_path, config, vocab)

    tf.set_random_seed(111)

    # Setup training
    sess = tf.Session()

    # Feed forward test
    with sess:
        sess.run(...)
        output_shape = ...
        print('Feed forward OK! Output shape: %s' % str(output_shape))


    # Training - to comment while testing the feed forward pass
    with sess:
        summary_writer = setup_training(train_dir)

        # Run training
        tf.logging.info('Starting training...')
        while batcher.epoch_completed < config.epochs:
            batch = batcher.next_batch()
            iteration = model.run_train_step(sess, batch)

            loss = iteration['loss']
            tf.logging.info('Loss: %5.3f' % loss)

            summaries = iteration['summary']

            summary_writer.add_summary(summaries, train_step)  # write the summaries
