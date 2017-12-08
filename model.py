# Defining the model
from typing import Dict, Tuple

import tensorflow as tf
import tensornets as nets
import numpy as np
from create_graph import CaptioningGraph

class CaptioningNetwork():
    def __init__(self, config, vocab):
        """
        :param config: dictionnary with hyperparameters of the model
        Initializes a variable with all the hyperparameters
        """

        self.hyps = config
        self.vocab = vocab
        for key, value in self.hyps.items():
            print(key, ":", value)

    def create_placeholders(self):
        """

        :return: Creates all the necessary placeholders for the training loop, as class variables
        """
        self.X_pl = tf.placeholder(tf.float32, shape=[None, self.hyps['im_width'], self.hyps['im_height'], self.hyps['nb_channels']], name='X_input')
        self.y_pl = tf.placeholder(tf.int32, shape=[None, self.hyps['max_sentence_length']], name='y_target_in')

    def make_feed_dict(self, batch: Tuple) -> Dict:
    # def make_feed_dict(self, graph, batch: Tuple) -> Dict:
        """

        :param batch: Tuple with images and associated annotations
        :return: A dictionnary with the right feed_dict structure for TensorFlow. Will be used in the training loop.
        """
        feed_dict = {}
        images, annotations = batch
        feed_dict[self.X_pl] = images
        feed_dict[self.y_pl] = annotations
        # #
        # feed_dict[graph.X_pl] = images
        # feed_dict[graph.y_pl] = annotations

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

        # Create graph from class
        # self.training_graph = CaptioningGraph(self.hyps, self.vocab, 'training')
        # self.validation_graph = CaptioningGraph(self.hyps, self.vocab, 'validation')


    def add_operators(self):
        # Creating the training operators (loss/optimizers etc)
        self.add_train_op()
        self.add_valid_op()
        self.summaries = tf.summary.merge_all()

    def add_model(self):
        """
        :return: Creates model (final variable from feed_dict). The output is to define the self.out_tensor
        object, which is the final prediction tensor used in the loss function
        """

        #We have to remember loading the weights for this layer
        xav_init = tf.contrib.layers.xavier_initializer(uniform=False)
        embedding_init = tf.truncated_normal_initializer(0, self.hyps['embedding_sdv'])
        # embedding_init = tf.random_uniform_initializer(minval=-1.0, maxval=1.0, dtype=tf.float32)

        if self.hyps['pretrained_embedding'] == False:
            self.X_embeddings = tf.get_variable('X_embeddings',
                                                shape=[self.vocab.size, self.hyps['embedding_size']],
                                                dtype=tf.float32,
                                                initializer=embedding_init)

        else:
            self.X_embeddings = tf.Variable(np.load('./embedding_captioning.npy'), trainable=False, dtype=tf.float32)

        # tf.summary.scalar('Embeddings', tf.reduce_sum(tf.square(self.X_embeddings)))
        # self.cnn = nets.Inception3(self.X_pl)
        self.cnn = nets.Inception3(self.X_pl, is_training=True, classes=1000)

        # tf.summary.scalar('Output cnn', tf.reduce_sum(tf.square(self.cnn)))

        CNN_lastlayer = self.cnn
        # Shape sortie: [batchsize, 1000]


        cnn_output = tf.contrib.layers.fully_connected(CNN_lastlayer, self.hyps['hidden_dim'], scope='cnn_output')
        # cnn_output has shape [batch_size, hidden_size]

        cnn_output = tf.expand_dims(cnn_output, axis=1)
        # cnn_output has shape [batch_size, 1, hidden_size]

        # tf.summary.scalar('Output cnn after fully connected', tf.reduce_sum(tf.square(cnn_output)))

        # self.y_pl has shape (batch_size, max_sentence_length)
        # caption_embedding has shape (batch_size, max_sentence_length, embedding_size)
        caption_embedding = self.embedding(self.y_pl)

        inputs = tf.concat([cnn_output, caption_embedding], axis=1)

        # CNN_lastlayer : [batch_size, hidden_dim]
        # caption_embedding : [batch_size, nb_LSTM_cells, embedding_size] where embedding_size = hidden_dim

        lstmcell = tf.contrib.rnn.LSTMCell(self.hyps['hidden_dim'])

        # outputs, state = tf.nn.dynamic_rnn(cell=lstmcell,  inputs=inputs, dtype=tf.float32)

        # CNN Input
        # _, state = lstmcell(cnn_output, (init_state, init_state))

        # Caption input
        outputs, _ = tf.nn.dynamic_rnn(cell=lstmcell,  inputs=inputs, dtype=tf.float32, initial_state=lstmcell.zero_state(tf.shape(cnn_output)[0], dtype=tf.float32))
        # outputs, _ = tf.nn.dynamic_rnn(cell=lstmcell,  inputs=caption_embedding, dtype=tf.float32, initial_state=state)
        print('Training outputs shape: ', outputs.get_shape())
        # outputs : [batchsize, nombre de mots, hidden_dim]
        out_tensor = []
        unstacked_outputs = tf.unstack(outputs, axis=1)

        W_out = tf.get_variable('W_out', [self.hyps['hidden_dim'], self.vocab.size], initializer=xav_init)
        b_out = tf.get_variable('b_out', [self.vocab.size])

        # for output_batch in unstacked_outputs:
        for output_batch in unstacked_outputs[1:]:
            out_tensor.append(tf.nn.xw_plus_b(output_batch, W_out, b_out))

        # vocab_dists = [tf.nn.softmax(s) for s in out_tensor]
        vocab_dists = out_tensor
        stacked_vocab_dists = tf.stack(vocab_dists, axis=1)

        self.out_tensor = stacked_vocab_dists

        print(self.out_tensor.get_shape())
        # tf.summary.tensor_summary('distrib', self.out_tensor)

        self.out_sentences = [tf.argmax(i, axis=1) for i in vocab_dists]


        # Inference / validation part of the graph
        input = cnn_output
        inferred = []

        print('input shape', input.get_shape())

        output, state = tf.nn.dynamic_rnn(cell=lstmcell, inputs=input, dtype=tf.float32)
        # init_state = tf.zeros([tf.shape(input)[0], self.hyps['hidden_dim']])



        print('output shape', output.get_shape())
        print('state shape', state)

        # Input the token <GO>
        # input has dimensions (1, embedding_size)
        input = self.embedding(2)
        input = tf.expand_dims(input, axis=0)
        # input has dimensions (1, 1, embedding_size)

        # Batch size expand (Adding <GO> for all elements of the batch)
        ## The hard challenge here is to imitate the shape of the incoming tensor cnn_output, which is None

        # input = tf.stack([input for i in tf.range(0, limit=tf.shape(cnn_output)[0])])
        # input = tf.tile(input, tf.constant([tf.shape(cnn_output)[0].value, 1]))
        # input = tf.stack([input for i in range(self.hyps['valid_batch_size'])])
        # input = tf.expand_dims(input, axis=1)

        ones = tf.ones_like(output)
        input = tf.multiply(ones, input)
        # input batch_size, 1, 100
        print('<GO> inputs: ', input.get_shape())

        valid_dists = []
        # input
        for steps in range(self.hyps['max_sentence_length']):

            output, state = tf.nn.dynamic_rnn(cell=lstmcell, inputs=input, initial_state=state)
            # print('loop shapes. input', input.get_shape(), 'state', state)

            distrib = tf.nn.xw_plus_b(tf.unstack(output, axis=1)[0], W_out, b_out)
            # distrib = tf.nn.softmax(distrib)
            valid_dists.append(distrib)
            index = tf.argmax(distrib, axis=1)
            inferred.append(index)
            input = self.embedding(index)
            input = tf.expand_dims(input, axis=1)

        self.out_tensor_valid = tf.stack(valid_dists, axis=1)
        self.inferred_sentence = inferred
        print('Out tensor shape: ', self.out_tensor_valid.get_shape())

    def calculate_loss(self, preds, y_pl):
        """
        :param preds: Prediction of the forward pass, as vocabulary distributions. Shape (batch_size, max_sentence_length, vocab_size)
        :param y_pl: True target. Shape (batch_size, sentence_length)
        :return: Likelihood of the prediction (defined in the paper). The  is an element of the graph (Tensor)
        """
        #
        temp = tf.one_hot(y_pl[:,1:], depth=self.vocab.size, axis=-1)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=temp, logits=preds[:,:-1, :])

        if self.hyps['masking_loss']:  # With masking
            zero = tf.constant(0, dtype=tf.int32)
            mask = tf.not_equal(y_pl, zero)[:, 1:]

            if self.hyps['normalize_loss']:
                lens = tf.count_nonzero(y_pl, axis=1, dtype=tf.float32)
                lens = tf.expand_dims(lens, axis=1)
                loss = tf.divide(loss, lens)

            loss = tf.boolean_mask(loss, mask)
            loss = tf.reduce_sum(loss)

        else:  # Without masking

            if self.hyps['normalize_loss']:
                lens = tf.count_nonzero(y_pl, axis=1, dtype=tf.float32)
                lens = tf.expand_dims(lens, axis=1)
                loss = tf.divide(loss, lens)

            loss = tf.reduce_sum(loss)

        loss = loss / self.hyps['batch_size']

        return loss

    def add_valid_op(self):
        self.valid_loss = self.calculate_loss(self.out_tensor_valid, self.y_pl)

        # self.valid_loss = self.calculate_loss(self.validation_graph.out_tensor_valid, self.validation_graph.y_pl)
        tf.summary.scalar('valid_loss', self.valid_loss)

    def add_train_op(self):
        """
        Assigns to a class variable the training operator (gradient iteration)
        """
        self.loss = self.calculate_loss(self.out_tensor, self.y_pl)
        # self.loss = self.calculate_loss(self.training_graph.out_tensor, self.training_graph.y_pl)
        tf.summary.scalar('train_loss', self.loss)

        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.hyps['sgd_learning_rate'])
        optimizer = tf.train.AdamOptimizer(learning_rate=self.hyps['adam_learning_rate'])
        # optimizer
        grads_and_vars = optimizer.compute_gradients(self.loss)


        # capped_gvs = [(tf.clip_by_value(grad, -self.hyps['grad_clip'], self.hyps['grad_clip']), var) for grad, var in grads_and_vars]

        self.grads = [tf.norm(i[0]) for i in grads_and_vars]


        self.vars = [tf.norm(i[1]) for i in grads_and_vars]

        for i, j in enumerate(self.grads):
            tf.summary.scalar('Gradient variable %i'%i, j)
            tf.summary.scalar('Variable %i' % i, self.vars[i])

        # for grad, var in grads_and_vars:
        #     tf.summary.scalar('grad %s' %var.name, grad)
        # tf.summary.scalar('grad', grads_and_vars[0])
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # self.learning_rate = optimizer._lr_t
        # tf.summary.scalar('learning_rate', optimizer._lr_t)

    def feed_forward_test(self, sess, batch):
        feed_dict = self.make_feed_dict(batch)
        fetches = {'output': self.out_tensor}

        # feed_dict = self.make_feed_dict(self.training_graph, batch)
        # fetches = {'output': self.training_graph.out_tensor}

        result = sess.run(fetches=fetches, feed_dict=feed_dict)
        print('Feed forward OK')
        return result

    def run_train_step(self, sess, batch: Tuple):
        """
        :param batch: Batch of images and annotations
        :return: run with a TF session a batch iteration, defining feed dict, fetches, etc.
        """

        feed_dict = self.make_feed_dict(batch)
        # feed_dict = self.make_feed_dict(self.training_graph, batch)
        fetches = {'loss': self.loss,
                   'train_op': self.train_op,
                   'summary': self.summaries,
                   'global_step': self.global_step,
                   'out_sentences': self.out_sentences,
                   'grads': self.grads,
                   'vars': self.vars
                   }

        result = sess.run(fetches=fetches, feed_dict=feed_dict)


        return result

    def run_valid_step(self, sess, valid_batch):
        fetches = {'inference': self.inferred_sentence,
                   'valid_loss': self.valid_loss,
                   'summary': self.summaries
                   }
        #
        # fetches = {'inference': self.validation_graph.inferred_sentence,
        #            'valid_loss': self.valid_loss,
        #            'summary': self.summaries
        #            }

        feed_dict = self.make_feed_dict(valid_batch)
        # feed_dict = self.make_feed_dict(self.validation_graph, valid_batch)

        result = sess.run(fetches=fetches, feed_dict=feed_dict)

        return result


    #
    # def predict(self, input):
    #     """
    #
    #     :param input: image to feed-forward
    #     :return: The generated caption
    #     """
    #     ##### WARNING - The graph has to be updated to be working with the inference mode (No target in the feed_dict)
    #
    #     fetches = [self.validation_graph.out_tensor]
    #     feed_dict = {self.validation_graph.X_pl: input}
    #
    #     with tf.Session() as sess:
    #         generated_caption = sess.run(fetches, feed_dict)
    #
    #     return generated_caption
