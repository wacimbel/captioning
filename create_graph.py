import tensorflow as tf
import tensornets as nets
import numpy as np

class CaptioningGraph:
    def __init__(self, hyps, vocab, mode='training'):
        self.mode = mode
        self.hyps = hyps
        self.vocab = vocab

        if self.mode == 'training':
            self.create_training_placeholders()

        elif self.mode == 'validation':
            self.create_valid_placeholders()
        else:
            tf.logging.debug('Wrong mode inputed in Captioning Graph class')

        self.create_graph()

    def create_training_placeholders(self):
        """

        :return: Creates all the necessary placeholders for the training loop, as class variables
        """
        self.X_pl = tf.placeholder(tf.float32, shape=[self.hyps['batch_size'], self.hyps['im_width'], self.hyps['im_height'],
                                                      self.hyps['nb_channels']], name='X_input')
        self.y_pl = tf.placeholder(tf.int32, shape=[self.hyps['batch_size'], self.hyps['max_sentence_length']], name='y_target_in')


    def create_valid_placeholders(self):
        """

        :return: Creates all the necessary placeholders for the training loop, as class variables
        """
        self.X_pl = tf.placeholder(tf.float32, shape=[self.hyps['valid_batch_size'], self.hyps['im_width'], self.hyps['im_height'],
                                                      self.hyps['nb_channels']], name='X_input')
        self.y_pl = tf.placeholder(tf.int32, shape=[self.hyps['valid_batch_size'], self.hyps['max_sentence_length']], name='y_target_in')

    def embedding(self, annotations):
        return tf.nn.embedding_lookup(self.X_embeddings, annotations)

    def create_graph(self):
        """
        :return: Creates model (final variable from feed_dict). The output is to define the self.out_tensor
        object, which is the final prediction tensor used in the loss function
        """

        # We have to remember loading the weights for this layer
        xav_init = tf.contrib.layers.xavier_initializer(uniform=False)
        # embedding_init = tf.initializers.truncated_normal(0, self.hyps['embedding_sdv'])
        embedding_init = tf.truncated_normal_initializer(0, self.hyps['embedding_sdv'])

        if self.hyps['pretrained_embedding'] == False:
            self.X_embeddings = tf.get_variable('X_embeddings',
                                                shape=[self.vocab.size, self.hyps['embedding_size']],
                                                dtype=tf.float32,
                                                initializer=embedding_init,
                                                reuse=True
                                                )

        else:
            self.X_embeddings = tf.Variable(np.load('./embedding_captioning.npy'), trainable=False,
                                            dtype=tf.float32)

        # tf.summary.scalar('Embeddings', tf.reduce_sum(tf.square(self.X_embeddings)))
        self.cnn = nets.Inception3(self.X_pl)

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

        outputs, state = tf.nn.dynamic_rnn(cell=lstmcell, inputs=inputs, dtype=tf.float32)

        # tf.summary.scalar('LSTM outputs', tf.reduce_sum(tf.square(outputs)))

        # outputs : [batchsize, nombre de mots, hidden_dim]
        out_tensor = []
        unstacked_outputs = tf.unstack(outputs, axis=1)

        W_out = tf.get_variable('W_out', [self.hyps['hidden_dim'], self.vocab.size], initializer=xav_init)
        b_out = tf.get_variable('b_out', [self.vocab.size])

        for output_batch in unstacked_outputs[1:]:
            out_tensor.append(tf.nn.xw_plus_b(output_batch, W_out, b_out))

        vocab_dists = [tf.nn.softmax(s) for s in out_tensor]

        stacked_vocab_dists = tf.stack(vocab_dists, axis=1)

        self.out_tensor = stacked_vocab_dists

        print(self.out_tensor.get_shape())
        # tf.summary.tensor_summary('distrib', self.out_tensor)

        self.out_sentences = [tf.argmax(i, axis=1) for i in vocab_dists]

        # Inference / validation part of the graph
        input = cnn_output
        inferred = []

        # print('input shape', input.get_shape())
        output, state = tf.nn.dynamic_rnn(cell=lstmcell, inputs=input, dtype=tf.float32)

        # print('output shape', output.get_shape())
        # print('state shape', state)

        # Input the token <GO>
        input = self.embedding(2)

        # Batch size expand (Adding <GO> for all elements of the batch)
        input = tf.stack([input for i in range(cnn_output.get_shape()[0])])

        # Sentence dimension expand
        input = tf.expand_dims(input, axis=1)

        # print('input valid shape', input.get_shape())

        valid_dists = []
        for steps in range(self.hyps['max_sentence_length']):
            output, state = tf.nn.dynamic_rnn(cell=lstmcell, inputs=input, initial_state=state)
            # print('loop shapes. input', input.get_shape(), 'state', state)

            distrib = tf.nn.xw_plus_b(tf.unstack(output, axis=1)[0], W_out, b_out)
            distrib = tf.nn.softmax(distrib)
            valid_dists.append(distrib)
            index = tf.argmax(distrib, axis=1)
            inferred.append(index)
            input = self.embedding(index)
            input = tf.expand_dims(input, axis=1)

        self.out_tensor_valid = tf.stack(valid_dists, axis=1)
        self.inferred_sentence = inferred
