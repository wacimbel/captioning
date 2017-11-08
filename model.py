# Defining the model
import tensorflow as tf
from typing import Dict, List, Tuple

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
        self.X_pl = tf.placeholder(tf.float32, shape=(self.hyps['batch_size'], self.hyps['im_width'], self.hyps['im_height'], self.hyps['nb_channels']), name='X_input')
        self.y_in_pl = tf.placeholder(tf.int32, shape=(self.hyps['batch_size'], self.hyps['max_sentence_length']), name='y_target_in')
        self.y_out_pl = tf.placeholder(tf.int32, shape=(self.hyps['batch_size'], self.hyps['max_sentence_length']), name='y_target_out')

    def make_feed_dict(self, batch: Tuple) -> Dict:
        """

        :param batch: Tuple with images and associated annotations
        :return: A dictionnary with the right feed_dict structure for TensorFlow. Will be used in the training loop.
        """
        feed_dict = {}
        images, annotations = batch
        feed_dict[self.X_pl] = images
        feed_dict[self.y_in_pl] = annotations
        feed_dict[self.y_out_pl] = annotations

        return feed_dict

    def build_graph(self):
        """
        :return: Creates, or returns, the graph corresponding to the model.
        """
        # Creating model
        self.add_model()

        # Creating the training operators (loss/optimizers etc)
        self.add_train_op()

    def add_model(self, CNN_lastlayer, caption_embedding):
        """
        :return: Creates model (final variable from feed_dict). The output is to define the self.out_tensor
        object, which is the final prediction tensor used in the loss function
        """
        rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
        # CNN_lastlayer : [batch_size, LSTM_dim]
        # caption_embedding : [batch_size, nb_LSTM_cells, embedding_size] where embedding_size = LSTM_dim

        lstmcell = tf.contrib.rnn.LSTMCell(self.hyps.LSTM_dim, initializer=rand_unif_init)
        outputs, state = tf.nn.dynamic_rnn(cell=lstmcell, initial_state=CNN_lastlayer, inputs=caption_embedding, dtype=tf.float32)

        # outputs : [batchsize, nombre de mots, LSTM_dim]

        W_out = tf.get_variable('W_out', [self.hyps.LSTM_dim, self.hyps.vocab_size])
        b_out = tf.get_variable('b_out', [self.hyps.vocab_size])
        
        out_tensor = []
        for output_batch in outputs:
            out_tensor.append(tf.matmul(output_batch, W_out) + b_out)
        
        vocab_dists = [tf.nn.softmax(s) for s in out_tensor]
        
        vocab_dists = tf.stack(vocab_dists)
        self.out_tensor = vocab_dists
            
    def calculate_loss(self, preds, y_out_pl):
        """
        :param preds: Prediction of the forward pass, as vocabulary distributions. Shape (batch_size, max_sentence_length, vocab_size)
        :param y_out_pl: True target. Shape (batch_size, sentence_length)
        :return: Likelihood of the prediction (defined in the paper). The loss is an element of the graph (Tensor)
        """


        temp = tf.reshape(y_out_pl, (self.hyps['batch_size'], self.hyps['max_sentence_length'], 1))

        unstacked_pred = tf.unstack(preds)
        unstacked_true = tf.unstack(temp)

        results = [tf.gather_nd(unstacked_pred[i],
                                tf.concat([tf.reshape(tf.range(0, self.hyps['max_sentence_length']), (-1, 1)), unstacked_true[i]], axis=1)) for i in
                   range(len(unstacked_pred))]
        results = -tf.log(tf.stack(results))

        return tf.reduce_sum(results)

    def add_train_op(self):
        """
        Assigns to a class variable the training operator (gradient iteration)
        """
        self.loss = self.calculate_loss(self.out_tensor, self.y_out_pl)
        optimizer = tf.train.GradientDescentOptimizer(self.hyps['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    def run_train_step(self, batch: Tuple):
        """
        :param batch: Batch of images and annotations
        :return: run with a TF session a batch iteration, defining feed dict, fetches, etc.
        """

    def predict(self, input):
        """

        :param input: image to feed-forward
        :return: The generated caption
        """

        # Feed forward with the trained model
        generated_caption = ...
        return generated_caption