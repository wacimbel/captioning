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
        self.X_pl = tf.placeholder(tf.float32, shape=[self.hyps['batch_size'], self.hyps['im_height'], self.hyps['im_width']], name='X_input')
        self.y_in_pl = tf.placeholder(tf.int32, shape=[self.hyps['batch_size'], self.hyps['vocab_size']], name='y_target')
        self.y_out_pl = tf.placeholder(tf.int32, shape=[self.hyps['batch_size'], self.hyps['vocab_size']], name='y_target')

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

    def loss_and_metric(self, preds: tf.Tensor):
        """
        Assigns to a class variable the training operator (gradient iteration)
        """

        loss = self.calculate_loss(self.y_out_pl, preds)
        optimizer = tf.train.GradientDescentOptimizer(self.hyps['learning_rate'])
        self.train_op = optimizer.apply_gradients(loss)

    def add_model(self):
        """
        :return: Creates model (final variable from feed_dict)
        """

    def calculate_loss(self, y_out_pl, preds):
        loss = 0
        for i, index in enumerate(y_out_pl):
            loss += -tf.log(tf.gather_nd(preds, (None, i, index)))

        return loss

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