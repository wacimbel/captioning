# Defining the model
import tensorflow as tf
from typing import Dict, List, Tuple
import numpy as np
import tensornets as nets

class CaptioningNetwork():
    def __init__(self, config):
        """
        :param config: dictionnary with hyperparameters of the model
        Initializes a variable with all the hyperparameters
        """

        self.hyps = config

    def create_placeholders(self):
        """

        :return: Creates all the necessary placeholders for the training loop, as class variables
        """
        self.X_pl = tf.placeholder(tf.float32, shape=[self.hyps['batch_size'], self.hyps['im_width'], self.hyps['im_height'], self.hyps['nb_channels']], name='X_input')
        self.y_pl = tf.placeholder(tf.int32, shape=[self.hyps['batch_size'], self.hyps['max_sentence_length']], name='y_target_in')

    def make_feed_dict(self, batch: Tuple) -> Dict:
        """

        :param batch: Tuple with images and associated annotations
        :return: A dictionnary with the right feed_dict structure for TensorFlow. Will be used in the training loop.
        """
        feed_dict = {}
        images, annotations = batch
        feed_dict[self.X_pl] = images
        feed_dict[self.y_pl] = annotations

        return feed_dict

    def embedding(self, annotations):

        self.X_embeddings = tf.get_variable('X_embeddings', shape=[self.hyps['vocab_size'], self.hyps['embedding_size']],
                                            dtype = tf.float32,
                                            initializer=tf.random_normal_initializer(stddev=self.hyps['embedding_sdv']))

        return tf.nn.embedding_lookup(self.X_embeddings, annotations)

    def build_graph(self):
        """
        :return: Creates, or returns, the graph corresponding to the model.
        """
        # Create placeholders
        self.create_placeholders()

        # Creating model
        self.add_model()

        # Creating the training operators (loss/optimizers etc)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.add_train_op()
        self.summaries = tf.summary.merge_all()

    def add_model(self):
        """
        :return: Creates model (final variable from feed_dict). The output is to define the self.out_tensor
        object, which is the final prediction tensor used in the loss function
        """
        #We have to remember loading the weights for this layer
        CNN_lastlayer = nets.Inception3(self.X_pl)
        # Shape sortie: [batchsize, 1000]

        W_transition = tf.get_variable('W_transition', [CNN_lastlayer.shape[1], self.hyps['hidden_dim']])
        b_transition = tf.get_variable('b_transition', [self.hyps['hidden_dim']])

        # rnn_input has shape [batch_size, hidden_size]
        cnn_output = tf.nn.xw_plus_b(CNN_lastlayer, W_transition, b_transition, name='rnn_input')
        cnn_output = tf.expand_dims(cnn_output, axis=1)

        # self.y_pl has shape (batch_size, max_sentence_length)
        # caption_embedding has shape (batch_size, max_sentence_length, embedding_size)
        caption_embedding = self.embedding(self.y_pl)
        inputs = tf.concat([cnn_output, caption_embedding], axis=1)

        rand_unif_init = tf.random_uniform_initializer(-self.hyps['rand_unif_init_mag'], self.hyps['rand_unif_init_mag'],
                                                       seed=123)
        # CNN_lastlayer : [batch_size, hidden_dim]
        # caption_embedding : [batch_size, nb_LSTM_cells, embedding_size] where embedding_size = hidden_dim


        lstmcell = tf.contrib.rnn.LSTMCell(self.hyps['hidden_dim'], initializer=rand_unif_init)
        outputs, state = tf.nn.dynamic_rnn(cell=lstmcell,  inputs=inputs, dtype=tf.float32)

        print(outputs)
        # outputs : [batchsize, nombre de mots, hidden_dim]

        W_out = tf.get_variable('W_out', [self.hyps['hidden_dim'], self.hyps['vocab_size']])
        b_out = tf.get_variable('b_out', [self.hyps['vocab_size']])

        out_tensor = []
        unstacked_outputs = tf.unstack(outputs, axis=1)

        for output_batch in unstacked_outputs[1:]:
            out_tensor.append(tf.matmul(output_batch, W_out) + b_out)

        # stacked_out_tensor = tf.stack(out_tensor, axis=1)
        vocab_dists = [tf.nn.softmax(s) for s in out_tensor]

        stacked_vocab_dists = tf.stack(vocab_dists, axis=1)
        self.out_tensor = stacked_vocab_dists

            
    def calculate_loss(self, preds, y_pl):
        """
        :param preds: Prediction of the forward pass, as vocabulary distributions. Shape (batch_size, max_sentence_length, vocab_size)
        :param y_pl: True target. Shape (batch_size, sentence_length)
        :return: Likelihood of the prediction (defined in the paper). The  is an element of the graph (Tensor)
        """

        temp = tf.reshape(y_pl, [self.hyps['batch_size'], self.hyps['max_sentence_length'], 1])

        unstacked_pred = tf.unstack(preds)
        unstacked_true = tf.unstack(temp)

        results = [tf.gather_nd(unstacked_pred[i],
                                tf.concat([tf.reshape(tf.range(0, self.hyps['max_sentence_length']), [-1, 1]), unstacked_true[i]], axis=1)) for i in
                   range(len(unstacked_pred))]
        results = -tf.log(tf.stack(results))

        return tf.reduce_sum(results)

    def add_train_op(self):
        """
        Assigns to a class variable the training operator (gradient iteration)
        """
        self.loss = self.calculate_loss(self.out_tensor, self.y_pl)
        optimizer = tf.train.GradientDescentOptimizer(self.hyps['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    def run_train_step(self, sess, batch: Tuple):
        """
        :param batch: Batch of images and annotations
        :return: run with a TF session a batch iteration, defining feed dict, fetches, etc.
        """

        feed_dict = self.make_feed_dict(batch)
        fetches = {'loss': self.loss,
                   'train_op': self.train_op,
                   'summary': self.summaries,
                   'global_step': self.global_step
                   }

        result = sess.run(fetches=fetches, feed_dict=feed_dict)

        return result

    def predict(self, input):
        """

        :param input: image to feed-forward
        :return: The generated caption
        """
        ##### WARNING - The graph has to be updated to be working with the inference mode (No target in the feed_dict)

        fetches = [self.out_tensor]
        feed_dict = {self.X_pl: input}

        with tf.Session() as sess:
            generated_caption = sess.run(fetches, feed_dict)

        return generated_caption