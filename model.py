# Defining the model
from typing import Dict, Tuple

import tensorflow as tf
import tensornets as nets
import numpy as np
class CaptioningNetwork():
    def __init__(self, config, vocab):
        """
        :param config: dictionnary with hyperparameters of the model
        Initializes a variable with all the hyperparameters
        """

        self.hyps = config
        self.vocab = vocab

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
        return tf.nn.embedding_lookup(self.X_embeddings, annotations)

    def build_graph(self):
        """
        :return: Creates, or returns, the graph corresponding to the model.
        """
        # Create placeholders
        self.create_placeholders()

        # Creating model
        self.add_model()

    def add_operators(self):
        # Creating the training operators (loss/optimizers etc)
        self.add_train_op()
        self.summaries = tf.summary.merge_all()

    def add_model(self):
        """
        :return: Creates model (final variable from feed_dict). The output is to define the self.out_tensor
        object, which is the final prediction tensor used in the loss function
        """
        #We have to remember loading the weights for this layer
        xav_init = tf.contrib.layers.xavier_initializer(uniform=False)
        embedding_init = tf.initializers.truncated_normal(0, self.hyps['embedding_sdv'])

        self.X_embeddings = tf.get_variable('X_embeddings',
                                            shape=[self.vocab.size, self.hyps['embedding_size']],
                                            dtype=tf.float32,
                                            initializer=embedding_init)

        self.cnn = nets.Inception3(self.X_pl)

        CNN_lastlayer = self.cnn
        # Shape sortie: [batchsize, 1000]

        cnn_output = tf.contrib.layers.fully_connected(CNN_lastlayer, self.hyps['hidden_dim'], scope='cnn_output')
        # cnn_output has shape [batch_size, hidden_size]
        cnn_output = tf.expand_dims(cnn_output, axis=1)
        # cnn_output has shape [batch_size, 1, hidden_size]

        # self.y_pl has shape (batch_size, max_sentence_length)
        # caption_embedding has shape (batch_size, max_sentence_length, embedding_size)
        caption_embedding = self.embedding(self.y_pl)
        inputs = tf.concat([cnn_output, caption_embedding], axis=1)

        # CNN_lastlayer : [batch_size, hidden_dim]
        # caption_embedding : [batch_size, nb_LSTM_cells, embedding_size] where embedding_size = hidden_dim

        lstmcell = tf.contrib.rnn.LSTMCell(self.hyps['hidden_dim'])

        outputs, state = tf.nn.dynamic_rnn(cell=lstmcell,  inputs=inputs, dtype=tf.float32)

        # outputs : [batchsize, nombre de mots, hidden_dim]
        out_tensor = []
        unstacked_outputs = tf.unstack(outputs, axis=1)

        W_out = tf.get_variable('W_out', [self.hyps['hidden_dim'], self.vocab.size], initializer=xav_init)
        b_out = tf.get_variable('b_out', [self.vocab.size])

        for output_batch in unstacked_outputs[1:]:
            out_tensor.append(tf.nn.xw_plus_b(output_batch, W_out, b_out))

        vocab_dists = [tf.nn.softmax(s) for s in out_tensor]


        for i, word in enumerate(vocab_dists):
            top2 = tf.nn.top_k(word, k=2)
            first_value = top2[0][0, 0]
            second_value = top2[0][0, 1]

            first_indices = top2[1][0, 0]
            second_indices = top2[1][0, 1]

            #Indices
            tf.summary.scalar('TOP1 word %d' % i, first_indices)
            tf.summary.scalar('TOP2 word %d' % i, second_indices)

            #Values
            tf.summary.scalar('TOP1 proba %d' % i, first_value)
            tf.summary.scalar('TOP2 proba %d' % i, second_value)


        stacked_vocab_dists = tf.stack(vocab_dists, axis=1)

        self.out_tensor = stacked_vocab_dists
        print(self.out_tensor.get_shape())
        tf.summary.tensor_summary('distrib', self.out_tensor)

        self.out_sentences = [tf.argmax(i, axis=1) for i in vocab_dists]


        # Inference part of the graph
        input = cnn_output
        inferred = []

        output, state = tf.nn.dynamic_rnn(cell=lstmcell, inputs=input, dtype=tf.float32)

        # Input the token <GO>
        input = self.embedding(2)
        # Batch size expand
        input = tf.stack([input for i in range(cnn_output.get_shape()[0])])
        # Sentence dimension expand
        input = tf.expand_dims(input, axis=1)

        for steps in range(self.hyps['max_sentence_length']):

            output, state = tf.nn.dynamic_rnn(cell=lstmcell, inputs=input, initial_state=state)

            distrib = tf.nn.xw_plus_b(tf.unstack(output, axis=1)[0], W_out, b_out)
            distrib = tf.nn.softmax(distrib)
            index = tf.argmax(distrib, axis=1)
            inferred.append(index)
            input = self.embedding(index)
            input = tf.expand_dims(input, axis=1)

        self.inferred_tensor = inferred



            
    def calculate_loss(self, preds, y_pl):
        """
        :param preds: Prediction of the forward pass, as vocabulary distributions. Shape (batch_size, max_sentence_length, vocab_size)
        :param y_pl: True target. Shape (batch_size, sentence_length)
        :return: Likelihood of the prediction (defined in the paper). The  is an element of the graph (Tensor)
        """

        # temp = tf.reshape(y_pl, [self.hyps['batch_size'], self.hyps['max_sentence_length'], 1])
        #
        # unstacked_pred = tf.unstack(preds)
        # unstacked_true = tf.unstack(temp)
        #
        # results = [tf.gather_nd(unstacked_pred[i],
        #                         tf.concat([tf.reshape(tf.range(0, self.hyps['max_sentence_length']), [-1, 1]), unstacked_true[i]], axis=1)) for i in
        #            range(len(unstacked_pred))]
        #
        # results = -tf.log(tf.stack(results))
        #

        temp = tf.one_hot(y_pl, depth=self.vocab.size, axis=-1)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=temp, logits=preds)
        loss = tf.reduce_sum(loss)

        tf.summary.scalar('loss', loss)

        return loss

    def add_train_op(self):
        """
        Assigns to a class variable the training operator (gradient iteration)
        """
        self.loss = self.calculate_loss(self.out_tensor, self.y_pl)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.hyps['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    def feed_forward_test(self, sess, batch):
        feed_dict = self.make_feed_dict(batch)
        fetches = {'output': self.out_tensor}

        result = sess.run(fetches=fetches, feed_dict=feed_dict)
        print('Feed forward OK')
        return result

    def run_train_step(self, sess, batch: Tuple):
        """
        :param batch: Batch of images and annotations
        :return: run with a TF session a batch iteration, defining feed dict, fetches, etc.
        """

        feed_dict = self.make_feed_dict(batch)
        fetches = {'loss': self.loss,
                   'train_op': self.train_op,
                   'summary': self.summaries,
                   'global_step': self.global_step,
                   'out_sentences': self.out_sentences
                   }

        result = sess.run(fetches=fetches, feed_dict=feed_dict)


        return result

    def run_valid_step(self, sess, valid_batch):
        fetches = {'inference': self.inferred_tensor
                   # 'summary': self.summaries
                   }

        feed_dict = {}
        feed_dict[self.X_pl] = valid_batch[0]

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