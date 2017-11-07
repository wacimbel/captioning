# Defining the model
import tensorflow as tf

class CaptioningNetwork():
    def __init__(self, config):
        """
        :param config: dictionnary with hyperparameters of the model
        Initializes a variable with all the hyperparameters
        """

        self.hyps = config
        self.model = self.create_network()

    def create_placeholders(self):
        self.X_pl = tf.placeholder(tf.float32, shape=[self.hyps['batch_size'], self.hyps['im_height'], self.hyps['im_width']], name='X_input')
        self.y_in_pl = tf.placeholder(tf.int32, shape=[self.hyps['batch_size'], self.hyps['vocab_size']], name='y_target')
        self.y_out_pl = tf.placeholder(tf.int32, shape=[self.hyps['batch_size'], self.hyps['vocab_size']], name='y_target')

    def make_feed_dict(self, batch):
        feed_dict = {}
        images, annotations = batch
        feed_dict[self.X_pl] = images
        feed_dict[self.y_in_pl] = annotations
        feed_dict[self.y_out_pl] = annotations

        return feed_dict

    def build_graph(self):
        """
        :return:
        """

        self.create_model()
        self.add_train_op()

    def loss_and_metric(self, preds):
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

    def run_train_step(self, batch):
        """
        :param batch: Batch of images and annotations
        :return: run with a TF session a batch iteration, defining feed dict, fetches, etc.
        """

    def predict(self, input):
        """

        :param input: image to feed-forward
        :return: The generated caption
        """

        return generated_caption