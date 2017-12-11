# Training loop
import json
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import time
import tensornets as nets
from model import CaptioningNetwork
from data import Batcher, Vocab

# from tensorflow import word2
def setup_training(model, train_dir):
    """

    :param train_dir: Directory where the model and training checkpoints will be saved
    :return: A summary writer object, configured and ready to be used in the training loop
    """

    saver = tf.train.Saver(max_to_keep=3)  # keep 3 checkpoints at a time
    sv = tf.train.Supervisor(logdir=train_dir,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=10,  # save summaries for tensorboard every 60 secs
                             save_model_secs=60,  # checkpoint every 60 secs
                             global_step=model.global_step)

    summary_writer = sv.summary_writer

    return summary_writer


def run_and_print_validation(iteration_id, model, batcher, vocab):
    # valid_batch = batcher.next_val_batch(model.cnn)
    valid_batch = batcher.next_train_batch(model.cnn)

    inferred = model.run_valid_step(sess, valid_batch)
    valid_loss = inferred['valid_loss']

    valid_words = np.array([list(i) for i in inferred['inference']])
    valid_sentences = np.transpose(valid_words)

    print('\n----- VALIDATION --- Iteration %d -- Valid loss: %5.2f' % (iteration_id, valid_loss))

    success_rate = 0
    for id, k in enumerate(valid_sentences):
        pred = [vocab.get_index_word(j) for j in k]
        true = [vocab.get_index_word(j) for j in valid_batch[1][id]]

        mask = len([i for i in true if i != '<PAD>'])

        success_rate += pred[:mask] == true[:mask]
        print(id, ' '.join(pred))
        print(id, ' '.join(true))
        # print('\n')

    print('Success rate: %d / %d' % (success_rate, len(valid_sentences)))

def print_training(iteration_id, iteration, batch, vocab):

    print('\n----- TRANING --- Iteration %d -- Train loss: %5.2f' % (iteration_id, loss))
    train_words = np.array([list(i) for i in iteration['out_sentences']])
    train_sentences = np.transpose(train_words)

    success_rate = 0
    print('Grads', {i: j for i,j in enumerate(iteration['grads'])})
    print('Vars', {i: j for i,j in enumerate(iteration['vars'])})
    print('Max grad', max(iteration['grads']))
    print('Max var', max(iteration['vars']))
    for id, k in enumerate(train_sentences):
        pred = [vocab.get_index_word(j) for j in k]
        true = [vocab.get_index_word(j) for j in batch[1][id]]

        mask = len([i for i in true if i != '<PAD>'])

        success_rate += pred[:mask] == true[1:mask+1]
        print(id, ('<GO> '+' '.join(pred)).replace('<PAD>', ''))
        print(id, ' '.join(true))
        # print('\n')

    print('Success rate: %d / %d' % (success_rate, len(train_sentences)))

if __name__ == "__main__":

    config = json.load(open('config.json', 'r'))
    #data_path = '/dev/shm/coco/'
    data_path = 'coco/'
    train_dir = 'summaries/Caption_training' + datetime.datetime.strftime(datetime.datetime.today(), '%d%m%Y%H%M%S')

    vocab = Vocab('vocab')
    model = CaptioningNetwork(config, vocab)

    batcher = Batcher(data_path, config, vocab)

    tf.set_random_seed(111)

    # Setup training
    sess = tf.Session()
    tf.logging.info('Building graph...')
    model.build_graph()

    # print(tf.GraphKeys.GLOBAL_VARIABLES)
    # print(tf.GraphKeys.TRAINABLE_VARIABLES)


    # Feed forward test
       # with sess:
    #     sess.run(...)
    #     output_shape = ...
    #     print('Feed forward OK! Output shape: %s' % str(output_shape))

    ### Temporary test with one iteration
    # Training - to comment while testing the feed forward pass

    variables = {j: i.name for j, i in enumerate(tf.trainable_variables())}
    json.dump(variables, open('variable_names.json', 'w'))

    with sess:

        nets.pretrained(model.cnn)
        model.add_operators()
        sess.run(tf.global_variables_initializer())

        summary_writer = setup_training(model, train_dir)
        print('Ready to feedforward')

        # Run training
        tf.logging.info('Starting training...')

        # model.feed_forward_test(sess, batcher.next_batch(model.cnn))
        i = 0
        while batcher.epoch_completed < 10:
            # while batcher.epoch_completed < config.epochs:
            i += 1
            train_batch = batcher.next_train_batch(model.cnn)

            tf.logging.info('running training step...')
            t0 = time.time()
            iteration = model.run_train_step(sess, train_batch)
            t1 = time.time()
            tf.logging.info('seconds for training step: %.3f', t1 - t0)

            loss = iteration['loss']
            tf.logging.info('Loss: %5.3f' % loss)

            summaries = iteration['summary']

            summary_writer.add_summary(summaries, iteration['global_step'])  # write the summaries

            print('Iteration %d - Train loss: ' % i, loss)
            ## Validation
            if not i % 10:


                print('\n\n###############################################################################################')
                print('------------------------------------ ITERATION %d ---------------------------------------------' % i)

                print_training(i, iteration, train_batch, vocab)

                run_and_print_validation(i, model, batcher, vocab)
#
