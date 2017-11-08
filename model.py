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
        self.y_in_pl = tf.placeholder(tf.int32, shape=[self.hyps['batch_size'], self.hyps['vocab_size']], name='y_target_in')
        self.y_out_pl = tf.placeholder(tf.int32, shape=[self.hyps['batch_size'], self.hyps['vocab_size']], name='y_target_out')

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
        # Creating placeholders
        self.create_placeholders()
        # Creating model
        self.add_model()
        # Creating the training operators (loss/optimizers etc)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.add_train_op()
        self.summaries = tf.summary.merge_all()

    def add_train_op(self):
        """
        Assigns to a class variable the training operator (gradient iteration)
        """

        self.loss = self.calculate_loss(self.out_tensor, self.y_out_pl)
        optimizer = tf.train.GradientDescentOptimizer(self.hyps['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    def add_model(self):
        """
        :return: Creates model (final variable from feed_dict). The output is to define the self.out_tensor
        object, which is the final prediction tensor used in the loss function
        """


















        self.out_tensor = ...

    def calculate_loss(self, preds, y_out_pl):
        """

        :param preds: Prediction of the forward pass
        :param y_out_pl: True target
        :return: Likelihood of the prediction (defined in the paper). The loss is an element of the graph (Tensor)
        """
        loss = 0
        for i, index in enumerate(y_out_pl):
            loss += -tf.log(tf.gather_nd(preds, (None, i, index)))

        return loss

    def run_train_step(self, sess, batch: Tuple):
        """
        :param sess: TensorFlow session
        :param batch: Batch of images and annotations
        :return: run with a TF session a batch iteration, defining feed dict, fetches, etc.
        """

        feed_dict = self.make_feed_dict(batch)
        fetches = {'loss': self.loss,
                   'train_op': self.train_op,
                   'summary': self.summaries,
                   'global_step': self.global_step}

        result = sess.run(fetches=fetches, feed_dict=feed_dict)

        return result

    def predict(self, input_image):
        """

        :param input: image to feed-forward
        :return: The generated caption
        """

        # Feed forward with the trained model
        generated_caption = ...
        fetches = [self.out_tensor]
        feed_dict = {self.X_pl: input_image}
        with tf.Session() as sess:
            generated_caption = sess.run(fetches, feed_dict)

        return generated_caption