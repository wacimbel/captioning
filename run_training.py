# Training loop
import json
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import tensornets as nets
from model import CaptioningNetwork
from data import Batcher, Vocab


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
    tf.logging.info('Building graph...')
    model.build_graph()

    # Feed forward test
       # with sess:
    #     sess.run(...)
    #     output_shape = ...
    #     print('Feed forward OK! Output shape: %s' % str(output_shape))

    ### Temporary test with one iteration
    # Training - to comment while testing the feed forward pass


    with sess:

        sess.run(tf.global_variables_initializer())

        nets.pretrained(model.cnn)
        model.add_operators()

        summary_writer = setup_training(train_dir)
        print('Ready to feedforward')


        # Run training
        tf.logging.info('Starting training...')

        model.feed_forward_test(sess, batcher.next_batch(model.cnn))
        # while batcher.epoch_completed < config.epochs:
        #     batch = batcher.next_batch()
        #
        #     tf.logging.info('running training step...')
        #     t0 = time.time()
        #     iteration = model.run_train_step(sess, batch)
        #     t1 = time.time()
        #     tf.logging.info('seconds for training step: %.3f', t1 - t0)
        #
        #     loss = iteration['loss']
        #     tf.logging.info('Loss: %5.3f' % loss)
        #
        #     summaries = iteration['summary']
        #
        #     summary_writer.add_summary(summaries, iteration['global_step'])  # write the summaries